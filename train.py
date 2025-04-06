# -----------------------------------------------------------------------------
# Optimized Financial Sentiment Analysis with BERT using PyTorch - DDP + ESC
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataset import ConcatDataset
from transformers import BertModel, BertTokenizer, BertConfig, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import random
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import functools
import multiprocessing
from contextlib import nullcontext
#import wandb


# Set random seeds for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# Setup distributed training
def setup_ddp(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL for CUDA operations
        rank=rank,
        world_size=world_size,
        init_method='env://'
    )
    
    # Set device for current process
    torch.cuda.set_device(rank)

# Cleanup distributed training
def cleanup_ddp():
    dist.destroy_process_group()

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # We'll return the text and label directly
        # Tokenization will happen in the collate function
        return {
            'text': text,
            'label': label
        }

# Define a faster batch collate function
def collate_fn(batch, tokenizer, max_length):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Tokenize the entire batch at once (much faster)
    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'token_type_ids': encodings['token_type_ids'],
        'labels': torch.tensor(labels, dtype=torch.long)
    }

# Define the custom BERT model with classification head
class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_prob=0.1):
        super(BERTSentimentClassifier, self).__init__()
        # Load pre-trained BERT model with optimized config
        config = BertConfig.from_pretrained(bert_model_name)
        # Use static weights for positional embeddings to speed up
        config.position_embedding_type = 'absolute'
        
        # NOTE: We're NOT using flash attention because it would require a model re-initialization
        # which would lose the pre-trained weights. Instead, we'll rely on DDP and mixed precision
        # for performance improvements
        self.bert = BertModel.from_pretrained(bert_model_name, config=config)
        
        # Get the hidden size from the BERT config
        hidden_size = self.bert.config.hidden_size
        # Create classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Pass inputs through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        # Pass through the classifier
        logits = self.classifier(pooled_output)
        
        return logits
    

# Early Stopping class for training
class EarlyStopping:
    """
    Early stopping to stop training when validation performance doesn't improve.
    
    Args:
        patience (int): How many epochs to wait after last improvement
        delta (float): Minimum change to qualify as an improvement
        mode (str): Either 'min' for metrics to minimize (like loss) or 'max' for metrics to maximize (like F1)
        verbose (bool): If True, prints a message when improvement is found or early stopping is triggered
    """
    def __init__(self, patience=5, delta=0.001, mode='max', verbose=True):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Set initial best score as worst possible
        self.best_score = float('-inf') if mode == 'max' else float('inf')
    
    def __call__(self, current_score):
        if self.mode == 'min':
            # For metrics like loss that we want to minimize
            score = -1.0 * current_score
        else:
            # For metrics like accuracy/F1 that we want to maximize
            score = current_score
            
        # First epoch
        if self.best_score is None:
            self.best_score = score
            return False
            
        # Check if improvement is significant
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
                return True
        else:
            if self.verbose:
                if self.mode == 'max':
                    print(f'Validation metric improved from {self.best_score:.4f} to {score:.4f}')
                else:
                    print(f'Validation metric improved from {-self.best_score:.4f} to {-score:.4f}')
            self.best_score = score
            self.counter = 0
            
        return False
    

# Function to load and prepare data with caching
def prepare_data(data_path=['train.csv', 'test.csv'], fraction=None, cache_dir='./data_cache'):
    """
    Load and prepare the dataset with caching for faster loading.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filenames
    train_cache = os.path.join(cache_dir, f'train_cache_{fraction}.pkl')
    test_cache = os.path.join(cache_dir, f'test_cache_{fraction}.pkl')
    
    # Check if cache exists
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("Loading data from cache...")
        df_train = pd.read_pickle(train_cache)
        df_test = pd.read_pickle(test_cache)
        print(f"Data loaded from cache successfully")
        return df_train, df_test
    
    # Start timing
    start = time.time()
    
    # Load data with optimized settings
    print(f"Loading data from {data_path}...")
    
    # Use multiple processes to load large CSV files faster
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        df_train = pd.concat(
            pool.map(
                functools.partial(pd.read_csv), 
                [data_path[0]]
            ), 
            ignore_index=True
        )
        df_test = pd.concat(
            pool.map(
                functools.partial(pd.read_csv), 
                [data_path[1]]
            ), 
            ignore_index=True
        )
    
    # Check if fraction is given
    if fraction:
        # Sample fraction of the data from each class
        df_train = df_train.groupby('label').apply(
            lambda x: x.sample(frac=fraction, random_state=42)
        ).reset_index(drop=True)
        
        df_test = df_test.groupby('label').apply(
            lambda x: x.sample(frac=fraction, random_state=42)
        ).reset_index(drop=True)
    
    # Cache the processed dataframes
    df_train.to_pickle(train_cache)
    df_test.to_pickle(test_cache)
    
    print(f"Data preparation completed in {time.time() - start:.2f} seconds")
    return df_train, df_test

# Function to create data loaders for distributed training
def create_data_loaders(train_df, val_df, tokenizer, batch_size=32, max_length=512, 
                        num_workers=4, rank=0, world_size=1):
    # Create datasets
    train_dataset = FinancialSentimentDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = FinancialSentimentDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True  # Drop the last incomplete batch for consistent batch sizes
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create faster custom collate function
    batch_collate_fn = functools.partial(collate_fn, tokenizer=tokenizer, max_length=max_length)
    
    # Create data loaders with optimized settings
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True,           # Drop the last incomplete batch for consistent batch sizes
        prefetch_factor=2,        # Prefetch batches
        collate_fn=batch_collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,        # Prefetch batches
        collate_fn=batch_collate_fn
    )
    
    return train_dataloader, val_dataloader, train_sampler, val_sampler

# Training function with multi-GPU support
# Training function with multi-GPU support and early stopping
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, 
                num_epochs=3, gradient_accumulation_steps=2, rank=0, world_size=1,
                train_sampler=None, val_sampler=None, use_amp=True,
                early_stopping_patience=5, early_stopping_delta=0.001, 
                early_stopping_metric='f1', early_stopping_mode='max'):
    """
    Train the model with early stopping based on validation metrics.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs to train for
        gradient_accumulation_steps: Number of steps to accumulate gradients
        rank: Rank of current process in distributed training
        world_size: Total number of processes in distributed training
        train_sampler: Distributed sampler for training data
        val_sampler: Distributed sampler for validation data
        use_amp: Whether to use automatic mixed precision
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        early_stopping_delta: Minimum change to qualify as improvement
        early_stopping_metric: Metric to monitor for early stopping ('loss', 'accuracy', 'f1')
        early_stopping_mode: 'min' for metrics to minimize (like loss), 'max' for metrics to maximize
    """
    # Set up mixed precision training
    scaler = torch.amp.GradScaler() if use_amp else None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # To store training history
    training_stats = []
    
    # For tracking best model
    best_val_f1 = 0
    best_model_path = f"./best_model_lowdata_{rank}.pt"
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        delta=early_stopping_delta,
        mode=early_stopping_mode,
        verbose=(rank == 0)  # Only print on main process
    )
    
    # Training loop
    total_train_time = 0
    
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n{'=' * 20} Epoch {epoch+1}/{num_epochs} {'=' * 20}\n")
        
        # Set epoch for distributed sampler
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if val_sampler:
            val_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        total_train_loss = 0
        train_steps = 0
        epoch_start = time.time()
        
        # Progress bar for training (only on main process)
        if rank == 0:
            train_progress_bar = tqdm(train_dataloader, desc="Training")
        else:
            train_progress_bar = train_dataloader
        
        optimizer.zero_grad()  # Zero gradients once before loop
        
        for step, batch in enumerate(train_progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device) if 'labels' in batch else batch['label'].to(device) if 'label' in batch else None
            
            # Context manager for mixed precision training
            amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
            
            # Forward pass with automatic mixed precision if enabled
            with amp_ctx:
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling if using mixed precision
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update parameters with gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                if use_amp:
                    # Clip gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update parameters and learning rate
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Without mixed precision
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Update statistics
            total_train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1
            
            # Update progress bar (only on main process)
            if rank == 0:
                train_progress_bar.set_postfix({'loss': f"{loss.item() * gradient_accumulation_steps:.4f}"})
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / train_steps
        epoch_train_time = time.time() - epoch_start
        total_train_time += epoch_train_time
        
        if rank == 0:
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Training time: {epoch_train_time:.2f} seconds")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        # No gradient calculation during validation
        with torch.no_grad():
            # Progress bar for validation (only on main process)
            if rank == 0:
                val_progress_bar = tqdm(val_dataloader, desc="Validation")
            else:
                val_progress_bar = val_dataloader
            
            for batch in val_progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass - mixed precision context for validation too
                amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
                with amp_ctx:
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                
                # Update statistics
                total_val_loss += loss.item()
                val_steps += 1
                
                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar (only on main process)
                if rank == 0:
                    val_progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Gather predictions and labels from all processes
        if world_size > 1:
            # Create tensors to hold all predictions and labels
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            
            # Convert lists to tensors
            all_preds_tensor = torch.tensor(all_preds, device=device)
            all_labels_tensor = torch.tensor(all_labels, device=device)
            
            # Gather from all processes
            dist.all_gather_object(gathered_preds, all_preds_tensor)
            dist.all_gather_object(gathered_labels, all_labels_tensor)
            
            # Flatten gathered lists (only on main process)
            if rank == 0:
                all_preds = np.concatenate([tensor.cpu().numpy() for tensor in gathered_preds])
                all_labels = np.concatenate([tensor.cpu().numpy() for tensor in gathered_labels])
        
        # Calculate metrics (only on main process)
        if rank == 0:
            if val_steps > 0:
                avg_val_loss = total_val_loss / val_steps

                # Calculate metrics
                accuracy = accuracy_score(all_labels, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

                print(f"Validation Results:")
                print(f"Loss: {avg_val_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")

                # Log to wandb
            #    # wandb.log({
            #         'epoch': epoch + 1,
            #    #     'train_loss': avg_train_loss,
            #         'val_loss': avg_val_loss,
            #         'val_accuracy': accuracy,
            #         'val_precision': precision,
            #         'val_recall': recall,
            #         'val_f1': f1,
            #         'epoch_time': epoch_train_time
            #     })

                # Save best model
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), best_model_path)
                    print(f"Saved best model with F1 Score: {f1:.4f}")
                
                # Check early stopping criterion based on selected metric
                early_stopping_value = None
                if early_stopping_metric == 'loss':
                    early_stopping_value = avg_val_loss
                elif early_stopping_metric == 'accuracy':
                    early_stopping_value = accuracy
                elif early_stopping_metric == 'f1':
                    early_stopping_value = f1
                elif early_stopping_metric == 'precision':
                    early_stopping_value = precision
                elif early_stopping_metric == 'recall':
                    early_stopping_value = recall
                
                # Call early stopping
                if early_stopping_value is not None:
                    if early_stopping(early_stopping_value):
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Break out of epoch loop
                        break
            else:
                print("No validation steps were executed.")
        
        # Synchronize processes to ensure all follow early stopping decision
        if world_size > 1:
            # Share early stopping flag
            if rank == 0:
                early_stop_flag = torch.tensor([1 if early_stopping.early_stop else 0], device=device)
            else:
                early_stop_flag = torch.tensor([0], device=device)
            
            # Broadcast early stopping decision from rank 0 to all processes
            dist.broadcast(early_stop_flag, src=0)
            
            # If rank 0 triggered early stopping, all processes should stop
            if early_stop_flag.item() == 1:
                if rank != 0:
                    print(f"Process {rank}: Early stopping triggered by rank 0")
                break
    
    if rank == 0:
        print(f"Total training time: {total_train_time:.2f} seconds")
    
    # Load best model for return (only main process needs to do this)
    if rank == 0:
        if os.path.exists(best_model_path):
            # Load state dict to the module directly if using DDP
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(best_model_path))
            else:
                model.load_state_dict(torch.load(best_model_path))
    
    return model, training_stats

# Prediction function for 3-class sentiment analysis
def predict_sentiment(model, tokenizer, texts, device, max_length=512, batch_size=32, use_amp=True):
    model.eval()
    results = []
    
    # Process in batches for faster prediction
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize all texts in the batch
        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        token_type_ids = encodings['token_type_ids'].to(device)
        
        # Get prediction
        with torch.no_grad():
            amp_ctx = torch.amp.GradScaler('cuda') if use_amp else nullcontext()
            with amp_ctx:
                outputs = model(input_ids, attention_mask, token_type_ids)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Process each prediction
        for j, text in enumerate(batch_texts):
            # Map prediction to sentiment class
            if predictions[j] == 0:
                sentiment = "Negative"
            elif predictions[j] == 1:
                sentiment = "Neutral"
            elif predictions[j] == 2:
                sentiment = "Positive"
            else:
                sentiment = "Unknown"  # Just in case
                
            results.append({
                'text': text,
                'sentiment': sentiment,
                'prediction': int(predictions[j]),
                'original_class': int(predictions[j]) + 1  # Original 1-5 class
            })
    
    return results

# Worker function for multi-process distributed training
# Worker function for multi-process distributed training
def run_worker(rank, world_size, args):
    # Initialize distributed process group
    setup_ddp(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    
    # Define hyperparameters - optimized values for multi-GPU
    bert_model_name = args.get('bert_model_name', "bert-base-uncased")
    max_length = args.get('max_length', 256)  # Reduced length for faster processing
    batch_size = args.get('batch_size', 32) 
    num_epochs = args.get('num_epochs', 3)
    learning_rate = args.get('learning_rate', 2e-5)
    weight_decay = args.get('weight_decay', 0.01)
    warmup_ratio = args.get('warmup_ratio', 0.1)
    gradient_accumulation_steps = args.get('gradient_accumulation_steps', 1)
    num_workers = args.get('num_workers', 4)
    use_amp = args.get('use_amp', True)
    
    # Early stopping parameters
    early_stopping_patience = args.get('early_stopping_patience', 5)
    early_stopping_delta = args.get('early_stopping_delta', 0.001)
    early_stopping_metric = args.get('early_stopping_metric', 'f1')
    early_stopping_mode = args.get('early_stopping_mode', 'max')
    
    if rank == 0:
        print(f"Running on rank {rank} of {world_size}")
        print(f"Hyperparameters:")
        print(f"- bert_model_name: {bert_model_name}")
        print(f"- max_length: {max_length}")
        print(f"- batch_size: {batch_size} (per GPU)")
        print(f"- effective batch size: {batch_size * gradient_accumulation_steps * world_size}")
        print(f"- num_epochs: {num_epochs}")
        print(f"- learning_rate: {learning_rate}")
        print(f"- weight_decay: {weight_decay}")
        print(f"- warmup_ratio: {warmup_ratio}")
        print(f"- gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"- num_workers: {num_workers}")
        print(f"- mixed precision: {use_amp}")
        print(f"Early Stopping:")
        print(f"- patience: {early_stopping_patience}")
        print(f"- delta: {early_stopping_delta}")
        print(f"- metric: {early_stopping_metric}")
        print(f"- mode: {early_stopping_mode}")

        # wandb.init(
        #     project="financial-sentiment",   # Change this to your WandB project name
        #     config=args,
        #     reinit=True
        # )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    # Prepare data (only rank 0 prints progress)
    if rank == 0:
        print("Loading and preparing data...")
    train_df, val_df = prepare_data(fraction=args.get('data_fraction', 1.0))
    num_classes = len(train_df['label'].unique())
    
    if rank == 0:
        print(f"Number of classes: {num_classes}")
        print(f"Label distribution in training data: {train_df['label'].value_counts()}")
    
    # Create data loaders
    if rank == 0:
        print("Creating data loaders...")
    train_dataloader, val_dataloader, train_sampler, val_sampler = create_data_loaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size
    )
    
    # Initialize model
    if rank == 0:
        print("Initializing model...")
    model = BERTSentimentClassifier(
        bert_model_name=bert_model_name,
        num_classes=num_classes
    )
    model.to(device)

    # Compile the model for better performance
    model = torch.compile(model)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    
    # Set up optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    
    # Set up learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Train model
    if rank == 0:
        print("Starting training...")
    model, training_stats = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        use_amp=use_amp,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode
    )
    
    # Save model (only on main process)
    if rank == 0:
        model_path = args.get('model_path', "./financial_sentiment_model_optimized")
        os.makedirs(model_path, exist_ok=True)
        
        # Get model without DDP wrapper
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Save model state
        torch.save(model_to_save.state_dict(), os.path.join(model_path, "model_state.pt"))
        # Save tokenizer
        tokenizer.save_pretrained(model_path)
        
        print(f"Model saved to {model_path}")
        
        # Test with some examples
        test_texts = [
            "The company exceeded market expectations with record profits.",
            "Shares tumbled 15% following disappointing quarterly results.",
            "New regulations could pose significant challenges to revenue growth."
        ]
        
        results = predict_sentiment(model_to_save, tokenizer, test_texts, device, use_amp=use_amp)
        
        print("\nSentiment predictions:")
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Sentiment: {result['sentiment']}")
            print()

      #  artifact = wandb.Artifact("bert-sentiment-model", type="model")
       # artifact.add_file(os.path.join(model_path, "model_state.pt"))
       # wandb.log_artifact(artifact)
    
    # Clean up distributed process group
    cleanup_ddp()

  #  if rank == 0:
  #      wandb.finish()

# Main function to run the complete pipeline with multi-GPU support
# Main function to run the complete pipeline with multi-GPU support
def main():
    # Record start time
    start_time = time.time()
    
    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Check GPU availability and count
    if not torch.cuda.is_available():
        print("No CUDA devices available. Running on CPU.")
        device = torch.device("cpu")
        world_size = 1
    else:
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} CUDA devices")
        
        # Print CUDA information
        for i in range(world_size):
            print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # Define hyperparameters
    args = {
        # Model and training parameters
        'bert_model_name': "bert-base-uncased",
        'max_length': 512,                 # Reduced for speed
        'batch_size': 64,                  # Batch size per GPU
        'num_epochs': 50,                # Maximum number of epochs (early stopping may reduce this)
        'learning_rate': 2e-5,             # Slightly adjusted for multi-GPU
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': 1,  # Reduced since we have multiple GPUs now
        'num_workers': 2,                  # Workers per GPU
        'data_fraction': 0.1,              # Use 50% of data for testing
        'model_path': "./financial_sentiment_model_multi_gpu_law_data",
        'use_amp': True,                   # Use mixed precision
        
        # Early stopping parameters
        'early_stopping_patience': 5,      # Number of epochs with no improvement after which training will be stopped
        'early_stopping_delta': 0.001,     # Minimum change to qualify as an improvement
        'early_stopping_metric': 'accuracy',     # Metric to monitor ('loss', 'accuracy', 'f1', 'precision', 'recall')
        'early_stopping_mode': 'max',      # Mode ('min' for loss, 'max' for accuracy/f1/etc.)
    }
    
    # Multi-GPU training
    if world_size > 1:
        print(f"Launching {world_size} distributed processes")
        # Use torch.multiprocessing to launch multiple processes
        import torch.multiprocessing as mp
        mp.spawn(
            run_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        print("Running on a single device")
        run_worker(0, 1, args)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
