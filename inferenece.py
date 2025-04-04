import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and sample data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train = df_train.groupby('label').apply(lambda x: x.sample(frac=0.05, random_state=42)).reset_index(drop=True)
df_test = df_test.groupby('label').apply(lambda x: x.sample(frac=0.05, random_state=42)).reset_index(drop=True)
data_train = df_train[['text', 'label']]
data_test = df_test[['text', 'label']]

# Initialize tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Initialize model and move to CUDA
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=5
)
model = model.to(device)

# Create a custom dataset for Amazon reviews
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Load model from saved checkpoint
model = RobertaForSequenceClassification.from_pretrained("output/checkpoint-4500", num_labels=5)
model = model.to(device)  # Move the loaded model to CUDA

# Create validation dataset and dataloader
val_dataset = AmazonReviewDataset(data_test, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Set the model to evaluation mode
model.eval()

# Initialize lists to store predictions and true labels
predictions = []
true_labels = []

# Iterate through the validation DataLoader
for batch in val_loader:
    # Move all tensor inputs to CUDA
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
    # Get the predicted labels
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    predictions.extend(preds)
    true_labels.extend(labels.cpu().numpy())

# Calculate the metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

accuracy = accuracy_score(true_labels, predictions)
recall = recall_score(true_labels, predictions, average='weighted')
precision = precision_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate a detailed classification report
report = classification_report(true_labels, predictions, output_dict=True)
print("\nClassification Report:")
print(classification_report(true_labels, predictions))

# Create a confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Create per-class metrics visualization
class_metrics = pd.DataFrame(report).T
class_metrics = class_metrics.drop('accuracy', errors='ignore')
class_metrics = class_metrics.drop('macro avg', errors='ignore')
class_metrics = class_metrics.drop('weighted avg', errors='ignore')
class_metrics = class_metrics[['precision', 'recall', 'f1-score']].head(5)  # Keep only the 5 classes

plt.figure(figsize=(12, 6))
class_metrics.plot(kind='bar', ylim=[0, 1])
plt.title('Performance Metrics per Class')
plt.xlabel('Class')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.savefig('class_metrics.png')
plt.close()

# Create ROC curve and AUC for multi-class
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Get the probabilities for each class
with torch.no_grad():
    all_inputs = torch.cat([batch['input_ids'].to(device) for batch in DataLoader(val_dataset, batch_size=len(val_dataset))])
    all_masks = torch.cat([batch['attention_mask'].to(device) for batch in DataLoader(val_dataset, batch_size=len(val_dataset))])
    all_outputs = model(all_inputs, attention_mask=all_masks)
    all_probs = torch.nn.functional.softmax(all_outputs.logits, dim=1).cpu().numpy()

# Binarize the labels for ROC curve
n_classes = 5
true_labels_bin = label_binarize(true_labels, classes=range(n_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.close()

# Create a prediction error analysis
error_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred]
if len(error_indices) > 0:
    error_data = []
    for idx in error_indices[:min(10, len(error_indices))]:  # Get first 10 errors
        text = data_test.iloc[idx]['text']
        true_label = data_test.iloc[idx]['label']
        pred_label = predictions[idx]
        error_data.append({
            'Text': text[:100] + '...' if len(text) > 100 else text,
            'True Label': true_label,
            'Predicted Label': pred_label
        })
    
    error_df = pd.DataFrame(error_data)
    print("\nSample Prediction Errors:")
    print(error_df)
    
    # Save error analysis to CSV
    error_df.to_csv('prediction_errors.csv', index=False)

print("\nAll visualizations have been saved to disk.")