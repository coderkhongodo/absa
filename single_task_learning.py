"""
Single Task Learning for Vietnamese Aspect-based Sentiment Analysis using PyTorch
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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

class AspectDataset(Dataset):
    """Dataset for aspect detection"""
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 115):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize and pad
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class SentimentDataset(Dataset):
    """Dataset for sentiment classification"""
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 115):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize and pad
        tokens = self.tokenizer.encode(text, max_length=self.max_length)
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
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
        
        # Add words to vocabulary (keep most frequent words)
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

class BiLSTMModel(nn.Module):
    """BiLSTM model for classification"""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 200,
                 num_classes: int = 8, dropout: float = 0.3, task_type: str = 'aspect'):
        super(BiLSTMModel, self).__init__()

        self.task_type = task_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(dropout)

        # Dense layers with batch normalization
        self.dense1 = nn.Linear(hidden_dim * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dense2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dense3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        if task_type == 'aspect':
            self.output = nn.Linear(32, num_classes)  # Sigmoid for multi-label
        else:
            self.output = nn.Linear(32, num_classes)  # Softmax for multi-class

    def forward(self, input_ids):
        # Embedding
        embedded = self.embedding(input_ids)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state
        # For bidirectional LSTM, concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # Dense layers with batch normalization
        x = F.relu(self.bn1(self.dense1(hidden)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.dense2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.dense3(x)))
        x = self.dropout(x)

        # Output
        output = self.output(x)

        if self.task_type == 'aspect':
            output = torch.sigmoid(output)  # Multi-label classification

        return output

class AspectSentimentAnalyzer:
    """Main class for aspect-based sentiment analysis"""
    
    def __init__(self, max_length: int = 115, embedding_dim: int = 128, hidden_dim: int = 200):
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tokenizer = SimpleTokenizer()
        self.aspect_model = None
        self.sentiment_models = {}
        self.aspect_names = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower', 'others']
        
    def prepare_aspect_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare labels for aspect detection (binary classification)"""
        aspect_labels = []

        for _, row in df.iterrows():
            labels = []
            for aspect in self.aspect_names:
                # Convert -1 to 0 (no aspect), and 0,1,2 to 1 (has aspect)
                if aspect in row and pd.notna(row[aspect]) and row[aspect] != -1:
                    labels.append(1)
                else:
                    labels.append(0)
            aspect_labels.append(labels)

        return np.array(aspect_labels, dtype=np.float32)
    
    def prepare_sentiment_labels(self, df: pd.DataFrame, aspect: str) -> Tuple[List[str], np.ndarray]:
        """Prepare data for sentiment classification of specific aspect"""
        texts = []
        labels = []

        for _, row in df.iterrows():
            if aspect in row and pd.notna(row[aspect]) and row[aspect] != -1:
                texts.append(row['processed_data'])
                # Labels: 0=negative, 1=neutral, 2=positive
                if row[aspect] == 0:
                    labels.append(0)  # Negative
                elif row[aspect] == 1:
                    labels.append(1)  # Neutral
                elif row[aspect] == 2:
                    labels.append(2)  # Positive
                else:
                    labels.append(1)  # Default to neutral

        print(f"  Found {len(texts)} samples for {aspect}")
        if len(texts) > 0:
            unique_labels = set(labels)
            print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        return texts, np.array(labels)
    
    def train_aspect_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """Train aspect detection model"""
        print("Training aspect detection model...")
        
        # Prepare data
        train_texts = train_df['processed_data'].tolist()
        val_texts = val_df['processed_data'].tolist()
        
        # Build vocabulary
        all_texts = train_texts + val_texts
        self.tokenizer.fit_on_texts(all_texts)
        
        # Prepare labels
        train_labels = self.prepare_aspect_labels(train_df)
        val_labels = self.prepare_aspect_labels(val_df)
        
        # Create datasets
        train_dataset = AspectDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = AspectDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.aspect_model = BiLSTMModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.aspect_names),
            task_type='aspect'
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.aspect_model.parameters(), lr=lr)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience = 5  # Increased patience
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.aspect_model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = self.aspect_model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.aspect_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = self.aspect_model(input_ids)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping with improvement threshold
            improvement_threshold = 0.001
            if val_loss < best_val_loss - improvement_threshold:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.aspect_model.state_dict(), 'models/single_task_aspect_model.pth')
                print(f"  → Model improved, saved checkpoint")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.aspect_model.load_state_dict(torch.load('models/single_task_aspect_model.pth'))
        print("Aspect model training completed!")
    
    def train_sentiment_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                             epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        """Train sentiment classification models for each aspect"""
        print("Training sentiment classification models...")
        
        for aspect in self.aspect_names:
            if aspect == 'others':  # Skip 'others' as it's binary
                continue
                
            print(f"\nTraining sentiment model for {aspect}...")
            
            # Prepare data for this aspect
            train_texts, train_labels = self.prepare_sentiment_labels(train_df, aspect)
            val_texts, val_labels = self.prepare_sentiment_labels(val_df, aspect)
            
            if len(train_texts) == 0:
                print(f"No data for aspect {aspect}, skipping...")
                continue
            
            # Create datasets
            train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer, self.max_length)
            val_dataset = SentimentDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = BiLSTMModel(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_classes=3,  # negative, neutral, positive
                task_type='sentiment'
            ).to(device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 7  # Increased patience for sentiment models
            patience_counter = 0

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['labels'].to(device)

                        outputs = model(input_ids)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                train_loss /= len(train_loader)
                val_loss /= len(val_loader)

                if epoch % 5 == 0 or epoch < 10:
                    print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Early stopping with improvement threshold
                improvement_threshold = 0.001
                if val_loss < best_val_loss - improvement_threshold:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'models/single_task_sentiment_{aspect}.pth')
                    if epoch % 5 == 0 or epoch < 10:
                        print(f"    → Model improved, saved checkpoint")
                else:
                    patience_counter += 1
                    if epoch % 5 == 0 or epoch < 10:
                        print(f"    → No improvement ({patience_counter}/{patience})")
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model and store
            model.load_state_dict(torch.load(f'models/single_task_sentiment_{aspect}.pth'))
            self.sentiment_models[aspect] = model
            
        print("Sentiment models training completed!")
    
    def save_evaluation_results(self, results: Dict, save_dir: str = 'evaluation_results'):
        """Save evaluation results to files"""
        os.makedirs(save_dir, exist_ok=True)

        # Save results as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f'evaluation_results_{timestamp}.json')

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json_results = convert_numpy(results)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        print(f"Evaluation results saved to: {results_file}")
        return results_file

    def plot_confusion_matrices(self, results: Dict, save_dir: str = 'evaluation_results'):
        """Plot and save confusion matrices"""
        os.makedirs(save_dir, exist_ok=True)

        # Plot aspect detection confusion matrices
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Aspect Detection - Confusion Matrices', fontsize=16)

        for i, aspect in enumerate(self.aspect_names):
            row = i // 4
            col = i % 4

            if aspect in results['aspect_detection']:
                cm = results['aspect_detection'][aspect]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['No Aspect', 'Has Aspect'],
                           yticklabels=['No Aspect', 'Has Aspect'],
                           ax=axes[row, col])
                axes[row, col].set_title(f'{aspect.upper()}')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')

        plt.tight_layout()
        aspect_cm_file = os.path.join(save_dir, 'aspect_confusion_matrices.png')
        plt.savefig(aspect_cm_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Plot sentiment classification confusion matrices
        sentiment_aspects = [a for a in self.aspect_names if a != 'others' and a in results['sentiment_classification']]
        if sentiment_aspects:
            n_aspects = len(sentiment_aspects)
            cols = 3
            rows = (n_aspects + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            fig.suptitle('Sentiment Classification - Confusion Matrices', fontsize=16)

            if rows == 1:
                axes = axes.reshape(1, -1)

            for i, aspect in enumerate(sentiment_aspects):
                row = i // cols
                col = i % cols

                cm = results['sentiment_classification'][aspect]['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                           xticklabels=['Negative', 'Neutral', 'Positive'],
                           yticklabels=['Negative', 'Neutral', 'Positive'],
                           ax=axes[row, col])
                axes[row, col].set_title(f'{aspect.upper()} Sentiment')
                axes[row, col].set_xlabel('Predicted')
                axes[row, col].set_ylabel('Actual')

            # Hide empty subplots
            for i in range(n_aspects, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')

            plt.tight_layout()
            sentiment_cm_file = os.path.join(save_dir, 'sentiment_confusion_matrices.png')
            plt.savefig(sentiment_cm_file, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Confusion matrices saved to: {save_dir}")

    def plot_performance_summary(self, results: Dict, save_dir: str = 'evaluation_results'):
        """Plot performance summary charts"""
        os.makedirs(save_dir, exist_ok=True)

        # Aspect detection performance
        aspect_metrics = []
        for aspect in self.aspect_names:
            if aspect in results['aspect_detection']:
                metrics = results['aspect_detection'][aspect]['metrics']
                aspect_metrics.append({
                    'Aspect': aspect.upper(),
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'Accuracy': metrics['accuracy']
                })

        if aspect_metrics:
            df_aspect = pd.DataFrame(aspect_metrics)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Aspect Detection Performance Summary', fontsize=16)

            # Precision
            axes[0, 0].bar(df_aspect['Aspect'], df_aspect['Precision'], color='skyblue')
            axes[0, 0].set_title('Precision by Aspect')
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Recall
            axes[0, 1].bar(df_aspect['Aspect'], df_aspect['Recall'], color='lightgreen')
            axes[0, 1].set_title('Recall by Aspect')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # F1-Score
            axes[1, 0].bar(df_aspect['Aspect'], df_aspect['F1-Score'], color='orange')
            axes[1, 0].set_title('F1-Score by Aspect')
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Accuracy
            axes[1, 1].bar(df_aspect['Aspect'], df_aspect['Accuracy'], color='pink')
            axes[1, 1].set_title('Accuracy by Aspect')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            aspect_perf_file = os.path.join(save_dir, 'aspect_performance_summary.png')
            plt.savefig(aspect_perf_file, dpi=300, bbox_inches='tight')
            plt.close()

        # Sentiment classification performance
        sentiment_metrics = []
        for aspect in self.aspect_names:
            if aspect != 'others' and aspect in results['sentiment_classification']:
                metrics = results['sentiment_classification'][aspect]['metrics']
                sentiment_metrics.append({
                    'Aspect': aspect.upper(),
                    'Accuracy': metrics['accuracy'],
                    'Macro F1': metrics['macro_f1'],
                    'Weighted F1': metrics['weighted_f1']
                })

        if sentiment_metrics:
            df_sentiment = pd.DataFrame(sentiment_metrics)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Sentiment Classification Performance Summary', fontsize=16)

            # Accuracy
            axes[0].bar(df_sentiment['Aspect'], df_sentiment['Accuracy'], color='lightblue')
            axes[0].set_title('Accuracy by Aspect')
            axes[0].set_ylabel('Accuracy')
            axes[0].tick_params(axis='x', rotation=45)

            # Macro F1
            axes[1].bar(df_sentiment['Aspect'], df_sentiment['Macro F1'], color='lightcoral')
            axes[1].set_title('Macro F1-Score by Aspect')
            axes[1].set_ylabel('Macro F1-Score')
            axes[1].tick_params(axis='x', rotation=45)

            # Weighted F1
            axes[2].bar(df_sentiment['Aspect'], df_sentiment['Weighted F1'], color='lightyellow')
            axes[2].set_title('Weighted F1-Score by Aspect')
            axes[2].set_ylabel('Weighted F1-Score')
            axes[2].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            sentiment_perf_file = os.path.join(save_dir, 'sentiment_performance_summary.png')
            plt.savefig(sentiment_perf_file, dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Performance summary charts saved to: {save_dir}")

    def evaluate_models(self, test_df: pd.DataFrame):
        """Evaluate both aspect and sentiment models with visualization"""
        print("Evaluating models...")

        # Initialize results dictionary
        results = {
            'aspect_detection': {},
            'sentiment_classification': {},
            'timestamp': datetime.now().isoformat()
        }

        # Evaluate aspect model
        print("\n" + "="*50)
        print("ASPECT DETECTION EVALUATION")
        print("="*50)

        test_texts = test_df['processed_data'].tolist()
        test_labels = self.prepare_aspect_labels(test_df)

        test_dataset = AspectDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.aspect_model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.aspect_model(input_ids)
                predictions = (outputs > 0.5).float()

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Evaluate each aspect
        for i, aspect in enumerate(self.aspect_names):
            y_true = all_labels[:, i]
            y_pred = all_predictions[:, i]

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            # Calculate precision, recall, f1 manually for binary classification
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results['aspect_detection'][aspect] = {
                'confusion_matrix': cm,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
            }

            print(f"\n{aspect.upper()}:")
            print(classification_report(y_true, y_pred))

        # Evaluate sentiment models
        print("\n" + "="*50)
        print("SENTIMENT CLASSIFICATION EVALUATION")
        print("="*50)

        for aspect in self.aspect_names:
            if aspect == 'others' or aspect not in self.sentiment_models:
                continue

            print(f"\n{aspect.upper()} SENTIMENT:")

            test_texts, test_labels = self.prepare_sentiment_labels(test_df, aspect)

            if len(test_texts) == 0:
                print("No test data available")
                continue

            test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer, self.max_length)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            model = self.sentiment_models[aspect]
            model.eval()

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids)
                    predictions = torch.argmax(outputs, dim=1)

                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            cm = confusion_matrix(all_labels, all_predictions)

            # Get classification report as dict
            report = classification_report(all_labels, all_predictions,
                                         target_names=['Negative', 'Neutral', 'Positive'],
                                         output_dict=True, zero_division=0)

            results['sentiment_classification'][aspect] = {
                'confusion_matrix': cm,
                'metrics': {
                    'accuracy': accuracy,
                    'macro_f1': report['macro avg']['f1-score'],
                    'weighted_f1': report['weighted avg']['f1-score']
                },
                'classification_report': report
            }

            # Print results
            unique_labels = len(set(all_labels))
            if unique_labels == 1:
                print(f"Only 1 class found in test data: {set(all_labels)}")
                print(f"Accuracy: {accuracy:.4f}")
            elif unique_labels == 2:
                print(classification_report(all_labels, all_predictions,
                                          target_names=['Class_0', 'Class_1']))
            else:
                print(classification_report(all_labels, all_predictions,
                                          target_names=['Negative', 'Neutral', 'Positive']))

        # Save results and create visualizations
        print("\n" + "="*50)
        print("SAVING EVALUATION RESULTS")
        print("="*50)

        results_file = self.save_evaluation_results(results)
        self.plot_confusion_matrices(results)
        self.plot_performance_summary(results)

        print(f"\nEvaluation completed! Results saved in 'evaluation_results/' directory")
        return results

def main():
    """Main function"""
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv('data_pr/processed_train.csv')
    val_df = pd.read_csv('data_pr/processed_val.csv')
    test_df = pd.read_csv('data_pr/processed_test.csv')
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {val_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize analyzer
    analyzer = AspectSentimentAnalyzer()
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    # Train aspect detection model
    analyzer.train_aspect_model(train_df, val_df, epochs=30, batch_size=64, lr=0.001)

    # Train sentiment classification models
    analyzer.train_sentiment_models(train_df, val_df, epochs=30, batch_size=64, lr=0.001)
    
    # Evaluate models
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    analyzer.evaluate_models(test_df)
    
    # Save tokenizer
    with open('models/single_task_tokenizer.pkl', 'wb') as f:
        pickle.dump(analyzer.tokenizer, f)
    
    print("\nTraining and evaluation completed!")
    print("Models saved in 'models/' directory")

if __name__ == "__main__":
    main()
