"""
Amazon Review Classifier Training Script with Distributed Training Support
and Checkpoint Loading/Resuming

This training script can be run both on a single GPU in debug mode,
and also in a larger training run with distributed data parallel (DDP).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To resume training from a checkpoint:
$ python train.py --resume=True

To run with DDP on 4 GPUs on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 8 GPUs across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py

(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import argparse
from contextlib import nullcontext
import inspect

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import autocast, GradScaler

from transformers import RobertaModel, RobertaTokenizerFast
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# default config values
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 10
eval_iters = 25
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
resume = True  # if True, resume from latest checkpoint in out_dir

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'amazon-reviews'
wandb_run_name = 'roberta-classifier'  # 'run' + str(time.time())

# data
train_file = 'train.csv'
test_file = 'test.csv'
gradient_accumulation_steps = 8  # used to simulate larger batch sizes
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_length = 512  # max sequence length

# model
num_labels = 5  # number of classes for classification
freeze_encoder = True  # freeze the RoBERTa encoder
dropout = 0.3  # dropout rate for classifier

# adamw optimizer
learning_rate = 2e-5  # max learning rate
max_iters = 10000  # total number of training iterations
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 500  # how many steps to warm up for
lr_decay_iters = 10000  # should be ~= max_iters per Chinchilla
min_lr = 2e-6  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
# helper functions
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument('--out_dir', type=str, default=out_dir)
    parser.add_argument('--eval_interval', type=int, default=eval_interval)
    parser.add_argument('--log_interval', type=int, default=log_interval)
    parser.add_argument('--eval_iters', type=int, default=eval_iters)
    parser.add_argument('--eval_only', action='store_true', default=eval_only)
    parser.add_argument('--always_save_checkpoint', action='store_true', default=always_save_checkpoint)
    parser.add_argument('--resume', action='store_true', default=resume)
    
    # wandb logging
    parser.add_argument('--wandb_log', action='store_true', default=wandb_log)
    parser.add_argument('--wandb_project', type=str, default=wandb_project)
    parser.add_argument('--wandb_run_name', type=str, default=wandb_run_name)
    
    # data
    parser.add_argument('--train_file', type=str, default=train_file)
    parser.add_argument('--test_file', type=str, default=test_file)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=gradient_accumulation_steps)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--max_length', type=int, default=max_length)
    
    # model
    parser.add_argument('--num_labels', type=int, default=num_labels)
    parser.add_argument('--freeze_encoder', action='store_true', default=freeze_encoder)
    parser.add_argument('--dropout', type=float, default=dropout)
    
    # optimizer
    parser.add_argument('--learning_rate', type=float, default=learning_rate)
    parser.add_argument('--max_iters', type=int, default=max_iters)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--beta1', type=float, default=beta1)
    parser.add_argument('--beta2', type=float, default=beta2)
    parser.add_argument('--grad_clip', type=float, default=grad_clip)
    
    # learning rate decay
    parser.add_argument('--decay_lr', action='store_true', default=decay_lr)
    parser.add_argument('--warmup_iters', type=int, default=warmup_iters)
    parser.add_argument('--lr_decay_iters', type=int, default=lr_decay_iters)
    parser.add_argument('--min_lr', type=float, default=min_lr)
    
    # DDP
    parser.add_argument('--backend', type=str, default=backend)
    
    # system
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--dtype', type=str, default=dtype)
    parser.add_argument('--compile', action='store_true', default=compile)

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

class MaxPoolRobertaClassifier(nn.Module):
    """
    A classifier using RoBERTa with max pooling.
    """
    def __init__(self, num_labels=5, freeze_encoder=True, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        
        # Freeze RoBERTa encoder if requested
        if freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Classifier head with batch normalization for stability
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )
        # n of trainable parameters and non trainable parameters
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.num_non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"Trainable parameters: {self.num_trainable_params:,}")
        print(f"Non-trainable parameters: {self.num_non_trainable_params:,}")
        print(f"Total parameters: {self.num_trainable_params + self.num_non_trainable_params:,}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask to avoid padding tokens.
            labels (torch.Tensor, optional): Labels for computing the loss.
        
        Returns:
            logits (torch.Tensor): Logits for each class.
            loss (torch.Tensor, optional): Loss if labels are provided.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, H]
        
        # Mask padding tokens for pooling
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_hidden = hidden_states.masked_fill(mask == 0, -1e9)
        
        # Max pooling across token dimension
        pooled_output = torch.max(masked_hidden, dim=1).values  # [B, H]
        
        # Feed to classifier head
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure the optimizer for training.
        """
        # Collect parameters that need optimization
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create optimizer groups with weight decay following GPT approach
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Create AdamW optimizer
        use_fused = (device_type == 'cuda') and ('fused' in inspect_optimizer(optim.AdamW))
        optimizer = optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused
        )
        
        return optimizer

    def estimate_mfu(self, batch_size, dt):
        """
        Estimate the model flops utilization (MFU)
        """
        # Rough estimate based on the number of parameters and operations
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return 0.0  # Placeholder - would need more specific calculation for RoBERTa

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class AmazonReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Return the encoded input and label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def inspect_optimizer(optimizer_class):
    """Utility to check what keyword arguments are supported by an optimizer."""
    import inspect
    sig = inspect.signature(optimizer_class.__init__)
    return [param.name for param in sig.parameters.values()]

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it, args):
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / (args.warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > args.lr_decay_iters:
        return args.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# Evaluate the model on validation data
@torch.no_grad()
def evaluate(model, val_loader, device, ctx, eval_iters=None):
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Limit evaluation to eval_iters batches if specified
    if eval_iters is not None:
        val_loader = [next(iter(val_loader)) for _ in range(min(eval_iters, len(val_loader)))]
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with ctx:
            logits, loss = model(input_ids, attention_mask, labels)
        
        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    val_loss = total_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_report = classification_report(all_labels, all_preds)
    
    model.train()
    return val_loss, val_accuracy, val_report

# Find the latest checkpoint in a directory
def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in a directory."""
    checkpoint_files = [
        os.path.join(checkpoint_dir, f) 
        for f in os.listdir(checkpoint_dir) 
        if f.endswith('.pt')
    ]
    if not checkpoint_files:
        return None
    
    # Sort by file modification time, newest first
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoint_files[0]

# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------

def main():
    # Parse arguments
    args = get_args()
    
    # Various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        args.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(args.device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # Scale down gradient accumulation to maintain effective batch size
        if args.gradient_accumulation_steps > 1:
            assert args.gradient_accumulation_steps % ddp_world_size == 0
            args.gradient_accumulation_steps //= ddp_world_size
    else:
        # If not ddp, we're running on a single gpu
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    
    # Calculate effective tokens per iteration
    tokens_per_iter = args.gradient_accumulation_steps * ddp_world_size * args.batch_size * args.max_length
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        os.makedirs(args.out_dir, exist_ok=True)
    
    # Set the random seed
    torch.manual_seed(1337 + seed_offset)
    
    # Enable TF32 precision
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Setup autocast for mixed precision
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load data
    if master_process:
        print("Loading data...")
    df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)
    
    # Adjust labels to be 0-indexed if needed
    if 0 not in df_train['label'].unique() and 1 in df_train['label'].unique():
        df_train['label'] = df_train['label'] - 1
        df_test['label'] = df_test['label'] - 1
        if master_process:
            print("Adjusted labels to be 0-indexed")
            print(f"Train labels: {sorted(df_train['label'].unique())}")
            print(f"Test labels: {sorted(df_test['label'].unique())}")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    # Create datasets and loaders
    train_dataset = AmazonReviewDataset(df_train, tokenizer, max_length=args.max_length)
    test_dataset = AmazonReviewDataset(df_test, tokenizer, max_length=args.max_length)
    
    if ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=False
        )
    else:
        train_sampler = None
        test_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    if master_process:
        print("Initializing model...")
    model = MaxPoolRobertaClassifier(
        num_labels=args.num_labels,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout
    )
    
    # Initialize training variables
    iter_num = 0
    best_val_loss = float('inf')
    
    # Initialize a GradScaler for mixed precision training
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Setup optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    # Check for existing checkpoint to resume from
    if args.resume and master_process:
        checkpoint_path = find_latest_checkpoint(args.out_dir)
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            
            # Update model state
            model.load_state_dict(checkpoint['model'])
            
            # Update optimizer state if available
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                # Move optimizer states to GPU if needed
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(args.device)
            
            # Update training progress
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num']
                print(f"Resuming from iteration {iter_num}")
            
            # Update best validation loss
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print(f"Previous best validation loss: {best_val_loss:.4f}")
            
            # Check if we need to update args from checkpoint
            if 'args' in checkpoint:
                checkpoint_args = checkpoint['args']
                # Only update non-critical args that wouldn't cause issues if changed
                for key in ['learning_rate', 'min_lr', 'warmup_iters', 'lr_decay_iters', 'eval_interval', 'log_interval']:
                    if key in checkpoint_args:
                        print(f"Restoring {key}={checkpoint_args[key]} from checkpoint")
                        setattr(args, key, checkpoint_args[key])
    
    # Move model to device
    model.to(args.device)
    
    # Compile the model
    if args.compile and hasattr(torch, 'compile'):
        if master_process:
            print("Compiling the model (this may take a moment)...")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0+
    
    # Wrap model into DDP container if needed
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    
    # Extract raw model for checkpointing (unwrapping DDP container if needed)
    raw_model = model.module if ddp else model
    
    # Setup WandB logging
    if args.wandb_log and master_process:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        if iter_num > 0:  # If resuming, log that in wandb
            wandb.log({"resumed_from_iter": iter_num})
    
    # Do an initial evaluation
    if master_process:
        print("Running evaluation...")
        val_loss, val_accuracy, val_report = evaluate(
            model, test_loader, args.device, ctx, args.eval_iters
        )
        print(f"Initial val loss: {val_loss:.4f}, val accuracy: {val_accuracy:.4f}")
        print(val_report)
        
        # Only update best_val_loss if we're not resuming, or if it's better than the loaded one
        if not args.resume or val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Exit early if eval_only is set
    if args.eval_only:
        if ddp:
            destroy_process_group()
        return
    
    # Prepare for training
    if master_process:
        print(f"Starting training from iteration {iter_num}...")
    
    # Get initial batch - important for performance to prefetch
    train_iter = iter(train_loader)
    try:
        batch = next(train_iter)
    except StopIteration:
        # Reset dataloader if needed
        if ddp:
            train_loader.sampler.set_epoch(iter_num // len(train_loader))
        train_iter = iter(train_loader)
        batch = next(train_iter)
    
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_loss = 0.0
    
    # Main training loop
    while iter_num < args.max_iters:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, args) if args.decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 and master_process:
            val_loss, val_accuracy, val_report = evaluate(
                model, test_loader, args.device, ctx, args.eval_iters
            )
            
            print(f"Step {iter_num}: train loss {running_loss/args.log_interval:.4f}, "
                  f"val loss {val_loss:.4f}, val accuracy {val_accuracy:.4f}")
            print(val_report)
            
            # Log metrics if using WandB
            if args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": running_loss/args.log_interval if iter_num > 0 else 0.0,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                    "lr": lr,
                })
            running_loss = 0.0
            
            # Save a checkpoint if validation loss is better or always_save_checkpoint is True
            if val_loss < best_val_loss or args.always_save_checkpoint:
                best_val_loss = val_loss
                
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': vars(args),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"Saving checkpoint to {args.out_dir}")
                    torch.save(checkpoint, os.path.join(args.out_dir, f'ckpt_{iter_num}.pt'))
                    # Also save as the latest checkpoint
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt_latest.pt'))
        
        # Forward backward update with gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            # Move batch data to device
            input_ids = batch['input_ids'].to(args.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(args.device, non_blocking=True)
            labels = batch['label'].to(args.device, non_blocking=True)
            
            # Manage DDP gradient sync
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == args.gradient_accumulation_steps - 1
                )
            
            # Forward pass - wrapped in autocast for mixed precision
            with ctx:
                logits, loss = model(input_ids, attention_mask, labels)
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps
            
            # Prefetch next batch while GPU is busy
            try:
                next_batch = next(train_iter)
            except StopIteration:
                # Reset dataloader if needed
                if ddp:
                    train_loader.sampler.set_epoch(iter_num // len(train_loader) + 1)
                train_iter = iter(train_loader)
                next_batch = next(train_iter)
            
            # Backward pass with gradient scaling for mixed precision
            scaler.scale(loss).backward()
            
            # Track running loss
            running_loss += loss.item() * args.gradient_accumulation_steps
        
        # Batch is now fully processed, update the batch for next iteration
        batch = next_batch
        
        # Clip gradients if needed
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
        
        # Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update()
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % args.log_interval == 0 and master_process:
            print(f"iter {iter_num}: loss {loss.item() * args.gradient_accumulation_steps:.4f}, "
                  f"time {dt*1000:.2f}ms")
        
        # Increment iteration counters
        iter_num += 1
        local_iter_num += 1
    
    # Final evaluation
    if master_process:
        print("Training completed. Running final evaluation...")
        val_loss, val_accuracy, val_report = evaluate(
            model, test_loader, args.device, ctx
        )