import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.preprocessing import label_binarize
import itertools

# Define the custom BERT model with classification head (same as in training code)
class BERTSentimentClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_prob=0.1):
        super(BERTSentimentClassifier, self).__init__()
        # Load pre-trained BERT model with optimized config
        config = BertConfig.from_pretrained(bert_model_name)
        # Use static weights for positional embeddings to speed up
        config.position_embedding_type = 'absolute'
        
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

class FinancialSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add label
        encoding['label'] = torch.tensor(label, dtype=torch.long)
        
        return encoding


def load_model(model_path, bert_model_name, num_classes, device):
    """Load the trained model from disk."""
    print(f"Loading model from {model_path}...")
    
    # Initialize model
    model = BERTSentimentClassifier(
        bert_model_name=bert_model_name,
        num_classes=num_classes
    )
    
    # Load state dict
    state_dict = torch.load(os.path.join(model_path, "best_model_rank_0.pt"), map_location=device)
    
    # Handle torch.compile() prefix (_orig_mod.)
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("Detected '_orig_mod.' prefix from torch.compile(). Removing prefix...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_key = k[len('_orig_mod.'):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Load the fixed state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    return model


def load_tokenizer(model_path, bert_model_name):
    """Load the tokenizer from disk or from pretrained."""
    try:
        # Try to load from saved model
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded from saved model.")
    except:
        # If not available, load pretrained
        print(f"Loading pretrained tokenizer for {bert_model_name}...")
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        print("Pretrained tokenizer loaded.")
    
    return tokenizer


def load_test_data(test_file_path, fraction=None, cache_dir='./data_cache'):
    """
    Load test data for evaluation.
    
    Args:
        test_file_path: Path to the test CSV file
        fraction: If provided, sample this fraction of data from each class
        cache_dir: Directory for caching data
    """
    # Generate cache filename with fraction info
    cache_name = f'test_cache_eval_{fraction}.pkl' if fraction else 'test_cache_eval.pkl'
    test_cache = os.path.join(cache_dir, cache_name)
    
    # Check if cache exists
    if os.path.exists(test_cache):
        print(f"Loading test data from cache (fraction={fraction})...")
        df_test = pd.read_pickle(test_cache)
        print(f"Test data loaded from cache successfully: {len(df_test)} rows")
    else:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load data
        print(f"Loading test data from {test_file_path}...")
        df_test = pd.read_csv(test_file_path)
        
        # Apply stratified sampling if fraction is provided
        if fraction and 0.0 < fraction < 1.0:
            print(f"Sampling {fraction*100}% of data from each class...")
            # Sample fraction of the data from each class
            df_test = df_test.groupby('label').apply(
                lambda x: x.sample(frac=fraction, random_state=42)
            ).reset_index(drop=True)
            print(f"After sampling: {len(df_test)} rows")
        
        # Cache the processed dataframe
        df_test.to_pickle(test_cache)
        print(f"Test data cached for future use")
    
    return df_test


def evaluate_model(model, test_dataloader, device, num_classes):
    """Evaluate the model on test data and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC and PR curves
    
    # No gradient calculation during evaluation
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    
    # Prepare for ROC curve (multi-class)
    # For each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Binarize the labels for ROC curve calculation
    y_test_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Precision-Recall curve
    precision_curve = {}
    recall_curve = {}
    avg_precision = {}
    
    for i in range(num_classes):
        precision_curve[i], recall_curve[i], _ = precision_recall_curve(y_test_bin[:, i], all_probs[:, i])
        avg_precision[i] = average_precision_score(y_test_bin[:, i], all_probs[:, i])
    
    # Micro-average precision-recall curve
    precision_curve["micro"], recall_curve["micro"], _ = precision_recall_curve(
        y_test_bin.ravel(), all_probs.ravel())
    avg_precision["micro"] = average_precision_score(y_test_bin.ravel(), all_probs.ravel())
    
    # Store all metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'class_report': class_report,
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'avg_precision': avg_precision
    }
    
    return metrics


def plot_confusion_matrix(conf_matrix, class_names, output_path=None):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(fpr, tpr, roc_auc, num_classes, output_path=None):
    """Plot ROC curves for each class and micro-average."""
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pr_curves(precision_curve, recall_curve, avg_precision, num_classes, output_path=None):
    """Plot Precision-Recall curves for each class and micro-average."""
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall_curve[i], precision_curve[i], color=color, lw=2,
                 label=f'PR curve (class {i}) (AP = {avg_precision[i]:.2f})')
    
    # Plot micro-average PR curve
    plt.plot(recall_curve["micro"], precision_curve["micro"],
             label=f'Micro-average PR curve (AP = {avg_precision["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_class_distribution(all_labels, predicted_labels, class_names, output_path=None):
    """Plot the distribution of true vs predicted labels."""
    plt.figure(figsize=(12, 6))
    
    # Frequency of actual classes
    plt.subplot(1, 2, 1)
    sns.countplot(x=all_labels)
    plt.title('True Class Distribution')
    plt.xlabel('Class')
    plt.xticks(range(len(class_names)), class_names)
    
    # Frequency of predicted classes
    plt.subplot(1, 2, 2)
    sns.countplot(x=predicted_labels)
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.xticks(range(len(class_names)), class_names)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_metrics(class_report, class_names, output_path=None):
    """Plot performance metrics (precision, recall, f1) for each class."""
    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {}
    
    for class_idx, class_name in enumerate(class_names):
        if str(class_idx) in class_report:
            class_metrics[class_name] = [class_report[str(class_idx)][metric] for metric in metrics]
    
    # Create DataFrame for plotting
    df_metrics = pd.DataFrame(class_metrics, index=metrics).T
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    df_metrics.plot(kind='bar', rot=45, figsize=(12, 8))
    plt.title('Performance Metrics by Class')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.legend(loc='best')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confidence_histogram(all_probs, all_preds, all_labels, output_path=None):
    """Plot histogram of prediction confidences, divided into correct and incorrect."""
    # Get the confidence of the predicted class for each sample
    confidences = np.array([all_probs[i, pred] for i, pred in enumerate(all_preds)])
    
    # Determine which predictions were correct
    correct = (all_preds == all_labels)
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram for correct predictions
    plt.hist(confidences[correct], bins=20, alpha=0.7, color='green', 
             label=f'Correct Predictions ({sum(correct)})')
    
    # Plot histogram for incorrect predictions
    plt.hist(confidences[~correct], bins=20, alpha=0.7, color='red', 
             label=f'Incorrect Predictions ({sum(~correct)})')
    
    plt.xlabel('Confidence (Probability)')
    plt.ylabel('Number of Predictions')
    plt.title('Distribution of Model Confidence')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'confidence_histogram.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_misclassification_examples(test_df, all_preds, all_labels, class_names, n_examples=5, output_path=None):
    """Plot examples of misclassifications with highest confidence."""
    # Find misclassifications
    misclassified = all_preds != all_labels
    misclassified_indices = np.where(misclassified)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    
    # Get a sample of misclassified examples
    n_examples = min(n_examples, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, size=n_examples, replace=False)
    
    # Create a figure
    plt.figure(figsize=(15, n_examples * 2))
    
    # Plot each example
    for i, idx in enumerate(sample_indices):
        text = test_df.iloc[idx]['text']
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        
        plt.subplot(n_examples, 1, i + 1)
        plt.text(0.01, 0.5, f"Text: {text}\n\nTrue: {class_names[true_label]}\nPredicted: {class_names[pred_label]}", 
                 fontsize=12, wrap=True)
        plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(os.path.join(output_path, 'misclassification_examples.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Financial Sentiment Analysis Model')
    parser.add_argument('--model_path', type=str, default='./saved_models',
                        help='Path to the saved model')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased',
                        help='Name of the BERT model used for training')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='Path to the test data file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--class_names', type=str, default='Negative,Neutral,Positive',
                        help='Comma-separated list of class names')
    parser.add_argument('--fraction', type=float, default=None,
                        help='Fraction of test data to use (sampled from each class)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse class names
    class_names = args.class_names.split(',')
    num_classes = len(class_names)
    print(f"Using {num_classes} classes: {class_names}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path, args.bert_model_name)
    
    # Load test data with optional fraction
    test_df = load_test_data(args.test_file, args.fraction)
    print(f"Test data shape: {test_df.shape}")
    
    # Create test dataset and dataloader
    test_dataset = FinancialSentimentDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    # Load model
    model = load_model(args.model_path, args.bert_model_name, num_classes, device)
    
    # Evaluate model
    start_time = time.time()
    print("Evaluating model on test data...")
    metrics = evaluate_model(model, test_dataloader, device, num_classes)
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Print summary metrics
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("="*40)
    print("\nClassification Report:")
    for class_idx, class_name in enumerate(class_names):
        if str(class_idx) in metrics['class_report']:
            class_metrics = metrics['class_report'][str(class_idx)]
            print(f"Class {class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1-score']:.4f}")
            print(f"  Support: {class_metrics['support']}")
    print("="*40)
    
    # Generate and save all plots
    print("\nGenerating plots...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(metrics['conf_matrix'], class_names, args.output_dir)
    
    # 2. ROC Curves
    plot_roc_curves(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], num_classes, args.output_dir)
    
    # 3. Precision-Recall Curves
    plot_pr_curves(metrics['precision_curve'], metrics['recall_curve'], 
                  metrics['avg_precision'], num_classes, args.output_dir)
    
    # 4. Class Distribution
    plot_class_distribution(metrics['all_labels'], metrics['all_preds'], 
                           class_names, args.output_dir)
    
    # 5. Performance Metrics by Class
    plot_performance_metrics(metrics['class_report'], class_names, args.output_dir)
    
    # 6. Confidence Histogram
    plot_confidence_histogram(metrics['all_probs'], metrics['all_preds'], 
                             metrics['all_labels'], args.output_dir)
    
    # 7. Misclassification Examples
    plot_misclassification_examples(test_df, metrics['all_preds'], 
                                   metrics['all_labels'], class_names, 
                                   n_examples=5, output_path=args.output_dir)
    
    print(f"All evaluation plots saved to {args.output_dir}")
    
    # Save metrics to a file
    with open(os.path.join(args.output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write("="*40 + "\n\n")
        f.write("Classification Report:\n")
        for class_idx, class_name in enumerate(class_names):
            if str(class_idx) in metrics['class_report']:
                class_metrics = metrics['class_report'][str(class_idx)]
                f.write(f"Class {class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n")
    
    print(f"Metrics summary saved to {os.path.join(args.output_dir, 'metrics_summary.txt')}")
    
    # Save full evaluation results as JSON
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'conf_matrix': metrics['conf_matrix'].tolist(),
        'class_report': metrics['class_report'],
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"Full evaluation results saved to {os.path.join(args.output_dir, 'evaluation_results.json')}")
    
    # Analyze most confused classes
    conf_matrix = metrics['conf_matrix']
    class_confusion = {}
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:  # Skip the diagonal (correct predictions)
                # True class i was predicted as class j
                class_confusion[(i, j)] = conf_matrix[i, j]
    
    # Sort by confusion count
    sorted_confusion = sorted(class_confusion.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost Confused Classes:")
    for (true_class, pred_class), count in sorted_confusion[:5]:  # Top 5 confusions
        print(f"True: {class_names[true_class]}, Predicted: {class_names[pred_class]}, Count: {count}")
    
    # Example predictions for manual inspection
    print("\nGenerating sample predictions for manual inspection...")
    
    # Sample 10 random examples
    sample_indices = np.random.choice(len(test_df), size=10, replace=False)
    
    sample_texts = [test_df.iloc[i]['text'] for i in sample_indices]
    sample_labels = [test_df.iloc[i]['label'] for i in sample_indices]
    
    # Tokenize
    encodings = tokenizer(
        sample_texts,
        add_special_tokens=True,
        max_length=args.max_length,
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
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
    
    # Print results
    print("\nSample Predictions:")
    print("="*80)
    for i in range(len(sample_texts)):
        text = sample_texts[i][:100] + "..." if len(sample_texts[i]) > 100 else sample_texts[i]
        true_label = sample_labels[i]
        pred_label = preds[i].item()
        confidence = probs[i, pred_label].item()
        
        print(f"Text: {text}")
        print(f"True label: {class_names[true_label]}")
        print(f"Predicted: {class_names[pred_label]} (confidence: {confidence:.4f})")
        print("-"*80)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()