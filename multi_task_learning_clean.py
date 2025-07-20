"""
Multi-Task Learning for Vietnamese Aspect-based Sentiment Analysis using PyTorch
Based on the original notebook architecture with shared BiLSTM and task-specific heads
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class MultiTaskDataset(Dataset):
    """Dataset for multi-task learning following the original notebook structure"""
    def __init__(self, texts: List[str], aspect_labels: np.ndarray, 
                 sentiment_labels: Dict[str, np.ndarray], tokenizer, max_length: int = 115):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Aspect names following the original notebook order
        self.aspect_names = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        aspect_labels = self.aspect_labels[idx]
        
        # Tokenize and pad
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        # Prepare sentiment labels for each aspect
        sentiment_labels = {}
        for aspect in self.aspect_names:
            if aspect in self.sentiment_labels and idx < len(self.sentiment_labels[aspect]):
                sentiment_labels[aspect] = torch.tensor(self.sentiment_labels[aspect][idx], dtype=torch.long)
            else:
                sentiment_labels[aspect] = torch.tensor(0, dtype=torch.long)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'aspect_labels': torch.tensor(aspect_labels, dtype=torch.float),
            'sentiment_labels': sentiment_labels
        }

class SimpleTokenizer:
    """Simple tokenizer for Vietnamese text"""
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
    
    def fit_on_texts(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            words = str(text).split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add words to vocabulary
        for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text: str, max_length: int = 115) -> List[int]:
        """Encode text to token ids"""
        words = str(text).split()
        tokens = []
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx['<UNK>'])
        
        # Pad or truncate
        if len(tokens) < max_length:
            tokens.extend([self.word_to_idx['<PAD>']] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
        
        return tokens

class MultiTaskBiLSTM(nn.Module):
    """Multi-task BiLSTM model following the original notebook architecture"""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 200, 
                 num_aspects: int = 8, dropout: float = 0.2):
        super(MultiTaskBiLSTM, self).__init__()
        
        # Shared layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
        # Shared dense layers
        self.dense1 = nn.Linear(hidden_dim * 2, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        
        # Aspect detection head (8 aspects with sigmoid)
        self.aspect_head = nn.Linear(32, num_aspects)
        
        # Sentiment classification heads
        self.sentiment_heads = nn.ModuleDict()
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        
        for aspect in sentiment_aspects:
            self.sentiment_heads[aspect] = nn.Linear(33, 4)  # aspect_prob(1) + shared_features(32) = 33
    
    def forward(self, input_ids):
        # Shared feature extraction
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Shared dense layers
        x = F.relu(self.dense1(hidden))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        shared_features = F.relu(self.dense3(x))
        
        # Aspect detection
        aspect_logits = self.aspect_head(shared_features)
        aspect_probs = torch.sigmoid(aspect_logits)
        
        # Sentiment classification for each aspect
        sentiment_outputs = {}
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        
        for i, aspect in enumerate(sentiment_aspects):
            aspect_prob = aspect_probs[:, i:i+1]
            combined_features = torch.cat([aspect_prob, shared_features], dim=1)
            sentiment_outputs[aspect] = self.sentiment_heads[aspect](combined_features)
        
        return aspect_probs, sentiment_outputs

class MultiTaskAspectSentimentAnalyzer:
    """Multi-task learning for aspect-based sentiment analysis"""
    
    def __init__(self, max_length: int = 115, embedding_dim: int = 128, hidden_dim: int = 200):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tokenizer = SimpleTokenizer()
        self.model = None
        self.aspect_names = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower', 'others']
        
    def prepare_aspect_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare labels for aspect detection"""
        aspect_labels = []
        
        for _, row in df.iterrows():
            labels = []
            for aspect in self.aspect_names:
                if aspect in row and pd.notna(row[aspect]) and row[aspect] != -1:
                    labels.append(1)
                else:
                    labels.append(0)
            aspect_labels.append(labels)
        
        return np.array(aspect_labels, dtype=np.float32)
    
    def prepare_sentiment_labels(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare sentiment labels for each aspect"""
        sentiment_labels = {}
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        
        for aspect in sentiment_aspects:
            labels = []
            for _, row in df.iterrows():
                if aspect in row and pd.notna(row[aspect]):
                    if row[aspect] == -1:
                        labels.append(0)  # No aspect
                    elif row[aspect] == 0:
                        labels.append(1)  # Negative
                    elif row[aspect] == 1:
                        labels.append(2)  # Neutral
                    elif row[aspect] == 2:
                        labels.append(3)  # Positive
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
            
            sentiment_labels[aspect] = np.array(labels, dtype=np.int64)
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  {aspect}: {dict(zip(unique, counts))}")
        
        return sentiment_labels

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   epochs: int = 30, batch_size: int = 64, lr: float = 0.001):
        """Train multi-task model"""
        print("Training multi-task model...")

        # Prepare data
        train_texts = train_df['processed_data'].tolist()
        val_texts = val_df['processed_data'].tolist()

        # Build vocabulary
        all_texts = train_texts + val_texts
        self.tokenizer.fit_on_texts(all_texts)

        # Prepare labels
        train_aspect_labels = self.prepare_aspect_labels(train_df)
        val_aspect_labels = self.prepare_aspect_labels(val_df)

        train_sentiment_labels = self.prepare_sentiment_labels(train_df)
        val_sentiment_labels = self.prepare_sentiment_labels(val_df)

        # Create datasets
        train_dataset = MultiTaskDataset(train_texts, train_aspect_labels,
                                       train_sentiment_labels, self.tokenizer, self.max_length)
        val_dataset = MultiTaskDataset(val_texts, val_aspect_labels,
                                     val_sentiment_labels, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = MultiTaskBiLSTM(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_aspects=len(self.aspect_names)
        ).to(device)

        # Loss functions
        aspect_criterion = nn.BCELoss()
        sentiment_criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        best_val_loss = float('inf')
        patience = 7
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_aspect_loss = 0
            train_sentiment_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                aspect_labels = batch['aspect_labels'].to(device)

                optimizer.zero_grad()

                # Forward pass
                aspect_probs, sentiment_outputs = self.model(input_ids)

                # Aspect loss
                aspect_loss = aspect_criterion(aspect_probs, aspect_labels)

                # Sentiment losses
                sentiment_loss = 0
                for aspect in sentiment_outputs.keys():
                    sentiment_targets = torch.stack([batch['sentiment_labels'][aspect][i]
                                                   for i in range(len(batch['sentiment_labels'][aspect]))]).to(device)
                    sentiment_loss += sentiment_criterion(sentiment_outputs[aspect], sentiment_targets)

                # Total loss
                total_loss = aspect_loss + 0.5 * sentiment_loss

                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()
                train_aspect_loss += aspect_loss.item()
                train_sentiment_loss += sentiment_loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            val_aspect_loss = 0
            val_sentiment_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    aspect_labels = batch['aspect_labels'].to(device)

                    aspect_probs, sentiment_outputs = self.model(input_ids)

                    aspect_loss = aspect_criterion(aspect_probs, aspect_labels)

                    sentiment_loss = 0
                    for aspect in sentiment_outputs.keys():
                        sentiment_targets = torch.stack([batch['sentiment_labels'][aspect][i]
                                                       for i in range(len(batch['sentiment_labels'][aspect]))]).to(device)
                        sentiment_loss += sentiment_criterion(sentiment_outputs[aspect], sentiment_targets)

                    total_loss = aspect_loss + 0.5 * sentiment_loss

                    val_loss += total_loss.item()
                    val_aspect_loss += aspect_loss.item()
                    val_sentiment_loss += sentiment_loss.item()

            # Average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/multi_task_model.pth')
                print(f"  → Model improved, saved checkpoint")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('models/multi_task_model.pth'))
        print("Multi-task model training completed!")

def main():
    """Main function"""
    # Create directories
    os.makedirs('models', exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv('data_pr/processed_train.csv')
    val_df = pd.read_csv('data_pr/processed_val.csv')
    test_df = pd.read_csv('data_pr/processed_test.csv')

    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Initialize analyzer
    analyzer = MultiTaskAspectSentimentAnalyzer()

    # Train model
    print("\n" + "="*50)
    print("MULTI-TASK TRAINING PHASE")
    print("="*50)

    analyzer.train_model(train_df, val_df, epochs=30, batch_size=64, lr=0.001)

    # Save tokenizer
    with open('models/multi_task_tokenizer.pkl', 'wb') as f:
        pickle.dump(analyzer.tokenizer, f)

    print("\nMulti-task training completed!")
    print("Model saved as 'models/multi_task_model.pth'")
    print("Tokenizer saved as 'models/multi_task_tokenizer.pkl'")

if __name__ == "__main__":
    main()

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   epochs: int = 30, batch_size: int = 64, lr: float = 0.001):
        """Train multi-task model"""
        print("Training multi-task model...")

        # Prepare data
        train_texts = train_df['processed_data'].tolist()
        val_texts = val_df['processed_data'].tolist()

        # Build vocabulary
        all_texts = train_texts + val_texts
        self.tokenizer.fit_on_texts(all_texts)

        # Prepare labels
        train_aspect_labels = self.prepare_aspect_labels(train_df)
        val_aspect_labels = self.prepare_aspect_labels(val_df)

        train_sentiment_labels = self.prepare_sentiment_labels(train_df)
        val_sentiment_labels = self.prepare_sentiment_labels(val_df)

        # Create datasets
        train_dataset = MultiTaskDataset(train_texts, train_aspect_labels,
                                       train_sentiment_labels, self.tokenizer, self.max_length)
        val_dataset = MultiTaskDataset(val_texts, val_aspect_labels,
                                     val_sentiment_labels, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = MultiTaskBiLSTM(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_aspects=len(self.aspect_names)
        ).to(device)

        # Loss functions
        aspect_criterion = nn.BCELoss()
        sentiment_criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        best_val_loss = float('inf')
        patience = 7
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_aspect_loss = 0
            train_sentiment_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                aspect_labels = batch['aspect_labels'].to(device)

                optimizer.zero_grad()

                # Forward pass
                aspect_probs, sentiment_outputs = self.model(input_ids)

                # Aspect loss
                aspect_loss = aspect_criterion(aspect_probs, aspect_labels)

                # Sentiment losses
                sentiment_loss = 0
                for aspect in sentiment_outputs.keys():
                    sentiment_targets = torch.stack([batch['sentiment_labels'][aspect][i]
                                                   for i in range(len(batch['sentiment_labels'][aspect]))]).to(device)
                    sentiment_loss += sentiment_criterion(sentiment_outputs[aspect], sentiment_targets)

                # Total loss
                total_loss = aspect_loss + 0.5 * sentiment_loss

                total_loss.backward()
                optimizer.step()

                train_loss += total_loss.item()
                train_aspect_loss += aspect_loss.item()
                train_sentiment_loss += sentiment_loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            val_aspect_loss = 0
            val_sentiment_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    aspect_labels = batch['aspect_labels'].to(device)

                    aspect_probs, sentiment_outputs = self.model(input_ids)

                    aspect_loss = aspect_criterion(aspect_probs, aspect_labels)

                    sentiment_loss = 0
                    for aspect in sentiment_outputs.keys():
                        sentiment_targets = torch.stack([batch['sentiment_labels'][aspect][i]
                                                       for i in range(len(batch['sentiment_labels'][aspect]))]).to(device)
                        sentiment_loss += sentiment_criterion(sentiment_outputs[aspect], sentiment_targets)

                    total_loss = aspect_loss + 0.5 * sentiment_loss

                    val_loss += total_loss.item()
                    val_aspect_loss += aspect_loss.item()
                    val_sentiment_loss += sentiment_loss.item()

            # Average losses
            train_loss /= len(train_loader)
            train_aspect_loss /= len(train_loader)
            train_sentiment_loss /= len(train_loader)

            val_loss /= len(val_loader)
            val_aspect_loss /= len(val_loader)
            val_sentiment_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train - Total: {train_loss:.4f}, Aspect: {train_aspect_loss:.4f}, Sentiment: {train_sentiment_loss:.4f}")
            print(f"  Val   - Total: {val_loss:.4f}, Aspect: {val_aspect_loss:.4f}, Sentiment: {val_sentiment_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_multitask_model.pth')
                print(f"  → Model improved, saved checkpoint")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('models/best_multitask_model.pth'))
        print("Multi-task model training completed!")

    def evaluate_model(self, test_df: pd.DataFrame):
        """Evaluate multi-task model"""
        print("Evaluating multi-task model...")

        # Prepare test data
        test_texts = test_df['processed_data'].tolist()
        test_aspect_labels = self.prepare_aspect_labels(test_df)
        test_sentiment_labels = self.prepare_sentiment_labels(test_df)

        test_dataset = MultiTaskDataset(test_texts, test_aspect_labels,
                                      test_sentiment_labels, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()

        # Collect predictions
        all_aspect_preds = []
        all_aspect_labels = []
        all_sentiment_preds = {aspect: [] for aspect in ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']}
        all_sentiment_labels = {aspect: [] for aspect in ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']}

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                aspect_labels = batch['aspect_labels'].to(device)

                # Forward pass
                aspect_probs, sentiment_outputs = self.model(input_ids)

                # Aspect predictions
                aspect_preds = (aspect_probs > 0.5).float()
                all_aspect_preds.extend(aspect_preds.cpu().numpy())
                all_aspect_labels.extend(aspect_labels.cpu().numpy())

                # Sentiment predictions
                for aspect in sentiment_outputs.keys():
                    sentiment_preds = torch.argmax(sentiment_outputs[aspect], dim=1)
                    all_sentiment_preds[aspect].extend(sentiment_preds.cpu().numpy())

                    # Get sentiment labels for this batch
                    batch_sentiment_labels = []
                    for i in range(input_ids.size(0)):
                        batch_sentiment_labels.append(batch['sentiment_labels'][aspect][i].item())
                    all_sentiment_labels[aspect].extend(batch_sentiment_labels)

        # Convert to numpy arrays
        all_aspect_preds = np.array(all_aspect_preds)
        all_aspect_labels = np.array(all_aspect_labels)

        # Evaluate aspect detection
        print("\n" + "="*50)
        print("ASPECT DETECTION EVALUATION")
        print("="*50)

        for i, aspect in enumerate(self.aspect_names):
            y_true = all_aspect_labels[:, i]
            y_pred = all_aspect_preds[:, i]

            accuracy = accuracy_score(y_true, y_pred)
            print(f"\n{aspect.upper()}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(classification_report(y_true, y_pred))

        # Evaluate sentiment classification
        print("\n" + "="*50)
        print("SENTIMENT CLASSIFICATION EVALUATION")
        print("="*50)

        for aspect in all_sentiment_preds.keys():
            print(f"\n{aspect.upper()} SENTIMENT:")

            y_true = np.array(all_sentiment_labels[aspect])
            y_pred = np.array(all_sentiment_preds[aspect])

            accuracy = accuracy_score(y_true, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            print(classification_report(y_true, y_pred,
                                      target_names=['No Aspect', 'Negative', 'Neutral', 'Positive']))

        print("\nMulti-task evaluation completed!")

def main():
    """Main function"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv('data_pr/processed_train.csv')
    val_df = pd.read_csv('data_pr/processed_val.csv')
    test_df = pd.read_csv('data_pr/processed_test.csv')

    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Initialize analyzer
    analyzer = MultiTaskAspectSentimentAnalyzer()

    # Train model
    print("\n" + "="*50)
    print("MULTI-TASK TRAINING PHASE")
    print("="*50)

    analyzer.train_model(train_df, val_df, epochs=30, batch_size=64, lr=0.001)

    # Save tokenizer
    with open('models/multitask_tokenizer.pkl', 'wb') as f:
        pickle.dump(analyzer.tokenizer, f)

    print("\nMulti-task training and evaluation completed!")
    print("Model saved as 'models/best_multitask_model.pth'")
    print("Tokenizer saved as 'models/multitask_tokenizer.pkl'")

if __name__ == "__main__":
    main()
