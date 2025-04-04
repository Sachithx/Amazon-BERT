import pandas as pd
import numpy as np
import os
import torch
import random
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    RobertaTokenizerFast, 
    RobertaForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Random seed set to {seed_value}")

set_seed(42)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Data loading and preprocessing
def load_data(train_path='train.csv', test_path='test.csv', sample_frac=0.05):
    logger.info("Loading datasets...")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Show dataset statistics
    logger.info(f"Train dataset: {len(df_train)} samples")
    logger.info(f"Test dataset: {len(df_test)} samples")
    logger.info(f"Label distribution in train: {df_train['label'].value_counts().to_dict()}")
    
    # Sample data for quicker iterations if needed
    if sample_frac < 1.0:
        logger.info(f"Sampling {sample_frac*100}% of data for development")
        df_train = df_train.groupby('label').apply(lambda x: x.sample(frac=sample_frac, random_state=42)).reset_index(drop=True)
        df_test = df_test.groupby('label').apply(lambda x: x.sample(frac=sample_frac, random_state=42)).reset_index(drop=True)
    
    data_train = df_train[['text', 'label']]
    data_test = df_test[['text', 'label']]
    
    logger.info(f"Using {len(data_train)} train samples and {len(data_test)} test samples")
    
    # Check for NaN values and clean data
    logger.info(f"NaN values in train text: {data_train['text'].isna().sum()}")
    logger.info(f"NaN values in test text: {data_test['text'].isna().sum()}")
    
    # Fill NaN values with empty string
    data_train['text'] = data_train['text'].fillna("")
    data_test['text'] = data_test['text'].fillna("")
    
    return data_train, data_test

data_train, data_test = load_data(sample_frac=0.05)

# Custom dataset for Amazon reviews
class AmazonReviewDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Log some statistics about text length to determine appropriate max_length
        text_lengths = [len(text.split()) for text in self.data['text']]
        logger.info(f"Text length stats: min={min(text_lengths)}, max={max(text_lengths)}, avg={sum(text_lengths)/len(text_lengths):.1f}")
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Enhanced metrics computation with per-class reporting
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    
    # Overall metrics
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    
    # Per-class metrics for logging
    class_report = classification_report(labels, pred, output_dict=True)
    
    # Log confusion matrix
    cm = confusion_matrix(labels, pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    return {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
    }

# Initialize tokenizer
logger.info("Initializing RoBERTa tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Initialize model
logger.info("Initializing RoBERTa model...")
def init_model(checkpoint_path=None, num_labels=5):
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = RobertaForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels)
    else:
        logger.info("Initializing new model from roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
    
    model = model.to(device)
    return model

# Check if checkpoint exists, otherwise initialize from pretrained
checkpoint_path = "output/checkpoint-4500"
model = init_model(checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None)

# Create datasets
train_dataset = AmazonReviewDataset(data_train, tokenizer)
val_dataset = AmazonReviewDataset(data_test, tokenizer)

# Set up training arguments
run_name = f"roberta-amazon-{datetime.now().strftime('%Y%m%d-%H%M')}"
output_dir = f"output/{run_name}"

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=run_name,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effectively increases batch size
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,  # Only keep the 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard",
    logging_dir=f"logs/{run_name}",
    logging_steps=100,
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
    dataloader_num_workers=4,
    disable_tqdm=False
)

# Implement early stopping
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# Train the model
logger.info("Starting training...")
train_result = trainer.train()

# Log training results
logger.info(f"Training results: {train_result}")
logger.info(f"Training metrics: {train_result.metrics}")

# Save final model
model_output_dir = f"{output_dir}/final-model"
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
logger.info(f"Model saved to {model_output_dir}")

# Evaluate on test set
logger.info("Evaluating on test set...")
eval_results = trainer.evaluate(eval_dataset=val_dataset)
logger.info(f"Evaluation results: {eval_results}")

# Create visualizations of model performance
def create_visualizations(trainer, dataset):
    logger.info("Creating performance visualizations...")
    
    # Generate predictions
    raw_predictions, _, _ = trainer.predict(dataset)
    predictions = np.argmax(raw_predictions, axis=1)
    true_labels = [item['labels'].item() for item in [dataset[i] for i in range(len(dataset))]]
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    
    # Create per-class metrics visualization
    report = classification_report(true_labels, predictions, output_dict=True)
    class_metrics = pd.DataFrame(report).T
    class_metrics = class_metrics.drop('accuracy', errors='ignore')
    class_metrics = class_metrics.drop('macro avg', errors='ignore')
    class_metrics = class_metrics.drop('weighted avg', errors='ignore')
    class_metrics = class_metrics[['precision', 'recall', 'f1-score']].head(5)
    
    plt.figure(figsize=(12, 6))
    class_metrics.plot(kind='bar', ylim=[0, 1])
    plt.title('Performance Metrics per Class')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_metrics.png")
    plt.close()
    
    # Error analysis
    error_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred]
    if len(error_indices) > 0:
        error_data = []
        for idx in error_indices[:min(10, len(error_indices))]:
            text = dataset.data.iloc[idx]['text']
            true_label = dataset.data.iloc[idx]['label']
            pred_label = predictions[idx]
            error_data.append({
                'Text': text[:100] + '...' if len(text) > 100 else text,
                'True Label': true_label,
                'Predicted Label': pred_label
            })
        
        error_df = pd.DataFrame(error_data)
        logger.info("\nSample Prediction Errors:")
        logger.info(error_df)
        
        # Save error analysis to CSV
        error_df.to_csv(f"{output_dir}/prediction_errors.csv", index=False)
    
    logger.info("Visualizations created and saved.")

# Create visualizations
create_visualizations(trainer, val_dataset)

logger.info("Training and evaluation complete!")