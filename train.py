import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizerFast
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define the classifier model
class MaxPoolRobertaClassifier(nn.Module):
    """
    A simple classifier using RoBERTa with max pooling.
    This initiates the RoBERTa model and adds a classifier head on top.
    The RoBERTa model is frozen to prevent training.
    The classifier head consists of a series of linear layers with ReLU activations and dropout for regularization.
    
    Args:
        num_labels (int): The number of output labels for the classification task.
    """
    def __init__(self, num_labels=5):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        # Freeze RoBERTa encoder
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        # classifier head with batch normalization for stability
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask to avoid padding tokens.
        
        Returns:
            torch.Tensor: Logits for each class.
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
        return logits

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
            'label': torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_model(model, train_loader, val_loader=None, epochs=5, learning_rate=2e-5):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_accuracy = 0.0
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss - check for nan values and shapes
            try:
                # Print shape information for debugging
                if torch.isnan(outputs).any():
                    print(f"Warning: outputs contain NaN values")
                
                # Make sure labels are in the correct range for the model's output classes
                if torch.max(labels) >= outputs.size(1):
                    print(f"Error: Labels out of range. Max label: {torch.max(labels).item()}, Output size: {outputs.size(1)}")
                    continue
                
                loss = criterion(outputs, labels)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print("Warning: NaN loss detected, skipping batch")
                    continue
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                continue
            
            # Track loss and predictions
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
        
        # Calculate training accuracy
        train_accuracy = accuracy_score(all_labels, all_preds)
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')
        
        # Validation if provided
        if val_loader:
            val_loss, val_accuracy, val_report = evaluate_model(model, val_loader, criterion, device)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
            print(val_report)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_roberta_classifier.pt')
                print(f'Model saved with accuracy: {val_accuracy:.4f}')
    
    return model

# Evaluation function
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(data_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_report = classification_report(all_labels, all_preds)
    
    return val_loss, val_accuracy, val_report

# Prediction function
def predict(model, text, tokenizer, device):
    # Tokenize the text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, 1)
    
    return preds.item()

# Main function to run the training
def main():
    # Load the data
    df_train = pd.read_csv('train.csv')  # Replace with your actual file paths
    df_test = pd.read_csv('test.csv')
    
    # Important: Ensure labels start from 0 and are consecutive integers
    print("Before label adjustment:")
    print(f"Train labels: {df_train['label'].unique()}")
    print(f"Test labels: {df_test['label'].unique()}")
    
    # If labels are 1-indexed (1-5) instead of 0-indexed (0-4), adjust them
    if 0 not in df_train['label'].unique() and 1 in df_train['label'].unique():
        df_train['label'] = df_train['label'] - 1
        df_test['label'] = df_test['label'] - 1
        print("After label adjustment:")
        print(f"Train labels: {df_train['label'].unique()}")
        print(f"Test labels: {df_test['label'].unique()}")
    
    # Initialize the tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    
    # Since max text length is 441, we can use a max_length of 512 (standard for RoBERTa)
    max_length = 512
    
    # Create datasets
    train_dataset = AmazonReviewDataset(df_train, tokenizer, max_length)
    test_dataset = AmazonReviewDataset(df_test, tokenizer, max_length)
    
    # Create data loaders
    batch_size = 16  # Adjust based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model
    model = MaxPoolRobertaClassifier(num_labels=5)  # 5 classes for Amazon reviews
    
    # Train the model
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=3,  # Adjust based on your needs
        learning_rate=2e-5
    )
    
    # Evaluate the model on test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_report = evaluate_model(trained_model, test_loader, criterion, device)
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
    print(test_report)

if __name__ == "__main__":
    main()