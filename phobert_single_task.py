"""
Optimized PhoBERT Single-Task Learning for Vietnamese Aspect-based Sentiment Analysis
Using pre-trained PhoBERT-base-v2 model with enhanced optimization:
- 30 epochs with early stopping (patience=7)
- Lower learning rate (1e-5) with warmup
- Reduced dropout (0.1) and fewer frozen layers
- Enhanced classification heads with layer normalization
- Optimized AdamW parameters
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import pickle
import json
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class PhoBERTDataset(Dataset):
    """Dataset for PhoBERT with proper Vietnamese text preprocessing"""
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize all texts for faster training
        print("Pre-tokenizing texts...")
        self.tokenized_texts = []
        for text in tqdm(texts, desc="Tokenizing"):
            text = self.preprocess_vietnamese_text(text)
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.tokenized_texts.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_texts[idx]['input_ids'],
            'attention_mask': self.tokenized_texts[idx]['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float if len(self.labels[idx].shape) > 0 else torch.long)
        }
    
    def preprocess_vietnamese_text(self, text) -> str:
        """Preprocess Vietnamese text for PhoBERT"""
        # Handle NaN values
        if pd.isna(text) or text is None:
            return ""

        # Convert to lowercase
        text = str(text).lower()

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)

        # Remove extra spaces
        text = text.strip()

        return text

class PhoBERTAspectClassifier(nn.Module):
    """Optimized PhoBERT-based aspect detection classifier"""
    def __init__(self, model_name: str = "vinai/phobert-base-v2", num_aspects: int = 8,
                 dropout: float = 0.1, freeze_layers: int = 6):
        super(PhoBERTAspectClassifier, self).__init__()

        self.phobert = AutoModel.from_pretrained(model_name)

        # Freeze fewer layers for better fine-tuning
        if freeze_layers > 0:
            for param in self.phobert.embeddings.parameters():
                param.requires_grad = False

            for i in range(freeze_layers):
                if i < len(self.phobert.encoder.layer):
                    for param in self.phobert.encoder.layer[i].parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Enhanced classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(self.phobert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_aspects)
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.phobert.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        # Get PhoBERT outputs
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation with layer norm
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.layer_norm(pooled_output)

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)

        return logits

class PhoBERTSentimentClassifier(nn.Module):
    """Optimized PhoBERT-based sentiment classification classifier"""
    def __init__(self, model_name: str = "vinai/phobert-base-v2", num_classes: int = 4,
                 dropout: float = 0.1, freeze_layers: int = 8):
        super(PhoBERTSentimentClassifier, self).__init__()

        self.phobert = AutoModel.from_pretrained(model_name)

        # Freeze fewer layers for better sentiment understanding
        if freeze_layers > 0:
            for param in self.phobert.embeddings.parameters():
                param.requires_grad = False

            for i in range(freeze_layers):
                if i < len(self.phobert.encoder.layer):
                    for param in self.phobert.encoder.layer[i].parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.phobert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.phobert.config.hidden_size)
    
    def forward(self, input_ids, attention_mask):
        # Get PhoBERT outputs
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation with layer norm
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.layer_norm(pooled_output)

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Classification
        logits = self.classifier(pooled_output)

        return logits

class PhoBERTAspectSentimentAnalyzer:
    """PhoBERT-based aspect-based sentiment analysis"""
    
    def __init__(self, model_name: str = "vinai/phobert-base-v2", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Models
        self.aspect_model = None
        self.sentiment_models = {}
        
        # Aspect names
        self.aspect_names = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower', 'others']
        
    def prepare_aspect_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare labels for aspect detection (multi-label binary classification)"""
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
    
    def prepare_sentiment_labels(self, df: pd.DataFrame, aspect: str) -> np.ndarray:
        """Prepare sentiment labels for a specific aspect"""
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
                    labels.append(0)  # Default to no aspect
            else:
                labels.append(0)  # No aspect mentioned
        
        return np.array(labels, dtype=np.int64)
    
    def train_aspect_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                          epochs: int = 30, batch_size: int = 16, lr: float = 1e-5):
        """Train optimized aspect detection model with early stopping"""
        print("Training optimized PhoBERT aspect detection model...")

        # Prepare data - using full dataset
        train_texts = train_df['processed_data'].tolist()
        val_texts = val_df['processed_data'].tolist()

        train_labels = self.prepare_aspect_labels(train_df)
        val_labels = self.prepare_aspect_labels(val_df)

        print(f"Training on full dataset: {len(train_texts)} samples")
        
        # Create datasets
        train_dataset = PhoBERTDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = PhoBERTDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
        
        # Initialize optimized model
        self.aspect_model = PhoBERTAspectClassifier(
            model_name=self.model_name,
            num_aspects=len(self.aspect_names),
            dropout=0.1,
            freeze_layers=6
        ).to(device)
        
        # Label smoothing for better convergence
        class LabelSmoothingBCELoss(nn.Module):
            def __init__(self, smoothing=0.1):
                super().__init__()
                self.smoothing = smoothing
                self.bce = nn.BCEWithLogitsLoss()

            def forward(self, pred, target):
                # Apply label smoothing
                target = target * (1 - self.smoothing) + 0.5 * self.smoothing
                return self.bce(pred, target)

        # Optimized loss and optimizer with faster convergence
        criterion = LabelSmoothingBCELoss(smoothing=0.1)
        optimizer = optim.AdamW(
            self.aspect_model.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warm restarts for faster convergence
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-7  # Minimum learning rate
        )
        
        # Enhanced training loop with early stopping
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        patience = 7  # Early stopping patience
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.aspect_model.train()
            train_loss = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                logits = self.aspect_model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                loss.backward()

                # Gradient clipping for stability and faster convergence
                torch.nn.utils.clip_grad_norm_(self.aspect_model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()  # Step scheduler after each batch for cosine annealing

                train_loss += loss.item()

                # Update progress bar with current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'loss': train_loss / (len(progress_bar.iterable) if hasattr(progress_bar, 'iterable') else 1),
                    'lr': f'{current_lr:.2e}'
                })
            
            # Validation
            self.aspect_model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    logits = self.aspect_model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
            
            # Average losses
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Quick validation F1 calculation for better early stopping
            val_f1_scores = []
            self.aspect_model.eval()
            with torch.no_grad():
                val_preds = []
                val_labels = []
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    logits = self.aspect_model(input_ids, attention_mask)
                    preds = torch.sigmoid(logits) > 0.5

                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                # Calculate average F1 score
                val_preds = np.array(val_preds)
                val_labels = np.array(val_labels)

                for i in range(len(self.aspect_names)):
                    y_true = val_labels[:, i]
                    y_pred = val_preds[:, i]

                    # Calculate F1
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    val_f1_scores.append(f1)

                avg_val_f1 = np.mean(val_f1_scores)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {avg_val_f1:.4f}")

            # Enhanced early stopping based on F1 score improvement
            if avg_val_f1 > best_val_f1 + 0.001:  # F1 improvement threshold
                best_val_f1 = avg_val_f1
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.aspect_model.state_dict(), 'models/phobert_aspect_model.pth')
                print(f"  → Model improved (Val F1: {avg_val_f1:.4f}), saved checkpoint")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
        
        # Load best model
        self.aspect_model.load_state_dict(torch.load('models/phobert_aspect_model.pth'))
        print("PhoBERT aspect model training completed!")
    
    def train_sentiment_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                             epochs: int = 30, batch_size: int = 16, lr: float = 1e-5):
        """Train optimized sentiment classification models for each aspect"""
        print("Training optimized PhoBERT sentiment classification models...")

        # Prepare data - using full dataset
        train_texts = train_df['processed_data'].tolist()
        val_texts = val_df['processed_data'].tolist()

        print(f"Training sentiment models on full dataset: {len(train_texts)} samples")

        # Train sentiment model for each aspect (excluding 'others')
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        
        for aspect in sentiment_aspects:
            print(f"\nTraining sentiment model for {aspect}...")
            
            # Prepare labels for this aspect
            train_labels = self.prepare_sentiment_labels(train_df, aspect)
            val_labels = self.prepare_sentiment_labels(val_df, aspect)
            
            # Create datasets
            train_dataset = PhoBERTDataset(train_texts, train_labels, self.tokenizer, self.max_length)
            val_dataset = PhoBERTDataset(val_texts, val_labels, self.tokenizer, self.max_length)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
            
            # Initialize optimized model
            model = PhoBERTSentimentClassifier(
                model_name=self.model_name,
                num_classes=4,
                dropout=0.1,
                freeze_layers=8
            ).to(device)
            
            # Optimized loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999)
            )

            # Cosine annealing scheduler for sentiment models
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=3,  # Shorter restart period for sentiment
                T_mult=2,
                eta_min=1e-7
            )
            
            # Enhanced training loop with early stopping
            best_val_loss = float('inf')
            patience = 7
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0

                progress_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}")

                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    
                    loss.backward()

                    # Gradient clipping for sentiment models
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()  # Step after each batch

                    train_loss += loss.item()

                    # Show learning rate in progress
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': train_loss / (len(progress_bar.iterable) if hasattr(progress_bar, 'iterable') else 1),
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Validation
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        
                        logits = model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                        
                        val_loss += loss.item()
                
                # Average losses
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)

                print(f"    Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

                # Enhanced early stopping
                if val_loss < best_val_loss - 0.0001:  # Smaller threshold
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'models/phobert_sentiment_{aspect}.pth')
                    print(f"      → Model improved (Val Loss: {val_loss:.4f}), saved checkpoint")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break
                
                scheduler.step()
            
            # Load best model and store
            model.load_state_dict(torch.load(f'models/phobert_sentiment_{aspect}.pth'))
            self.sentiment_models[aspect] = model
            
        print("PhoBERT sentiment models training completed!")

    def evaluate_models(self, test_df: pd.DataFrame):
        """Evaluate both aspect and sentiment models"""
        print("Evaluating PhoBERT models...")

        test_texts = test_df['processed_data'].tolist()

        # Evaluate aspect detection
        print("\n" + "="*50)
        print("ASPECT DETECTION EVALUATION")
        print("="*50)

        test_aspect_labels = self.prepare_aspect_labels(test_df)
        test_dataset = PhoBERTDataset(test_texts, test_aspect_labels, self.tokenizer, self.max_length)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.aspect_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = self.aspect_model(input_ids, attention_mask)
                preds = torch.sigmoid(logits) > 0.5

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        aspect_results = {}
        for i, aspect in enumerate(self.aspect_names):
            y_true = all_labels[:, i]
            y_pred = all_preds[:, i]

            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            # Calculate metrics
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                if len(np.unique(y_true)) == 1:
                    if y_true[0] == 0:
                        tn, fp, fn, tp = len(y_true), 0, 0, 0
                    else:
                        tn, fp, fn, tp = 0, 0, 0, len(y_true)
                else:
                    tn, fp, fn, tp = 0, 0, 0, 0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            aspect_results[aspect] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            print(f"\n{aspect.upper()}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")

        # Evaluate sentiment classification
        print("\n" + "="*50)
        print("SENTIMENT CLASSIFICATION EVALUATION")
        print("="*50)

        sentiment_results = {}
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']

        for aspect in sentiment_aspects:
            if aspect in self.sentiment_models:
                print(f"\n{aspect.upper()} SENTIMENT:")

                test_labels = self.prepare_sentiment_labels(test_df, aspect)
                test_dataset = PhoBERTDataset(test_texts, test_labels, self.tokenizer, self.max_length)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                model = self.sentiment_models[aspect]
                model.eval()

                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        logits = model(input_ids, attention_mask)
                        preds = torch.argmax(logits, dim=1)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                y_true = np.array(all_labels)
                y_pred = np.array(all_preds)

                accuracy = accuracy_score(y_true, y_pred)

                # Get classification report
                report = classification_report(y_true, y_pred,
                                             target_names=['No Aspect', 'Negative', 'Neutral', 'Positive'],
                                             labels=[0, 1, 2, 3],
                                             output_dict=True, zero_division=0)

                sentiment_results[aspect] = {
                    'accuracy': accuracy,
                    'macro_f1': report['macro avg']['f1-score'],
                    'weighted_f1': report['weighted avg']['f1-score'],
                    'classification_report': report
                }

                print(f"Accuracy: {accuracy:.4f}")
                print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
                print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred,
                                           target_names=['No Aspect', 'Negative', 'Neutral', 'Positive'],
                                           labels=[0, 1, 2, 3]))

        return aspect_results, sentiment_results

    def save_evaluation_results(self, aspect_results: Dict, sentiment_results: Dict,
                               save_dir: str = 'evaluation_results'):
        """Save evaluation results"""
        os.makedirs(save_dir, exist_ok=True)

        results = {
            'model_type': 'phobert_single_task',
            'model_name': self.model_name,
            'max_length': self.max_length,
            'aspect_detection': aspect_results,
            'sentiment_classification': sentiment_results,
            'timestamp': datetime.now().isoformat()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'{save_dir}/phobert_results_{timestamp}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nResults saved to: {results_file}")
        return results_file

    def load_models(self):
        """Load trained models"""
        print("Loading PhoBERT models...")

        # Load aspect model
        self.aspect_model = PhoBERTAspectClassifier(
            model_name=self.model_name,
            num_aspects=len(self.aspect_names)
        ).to(device)
        self.aspect_model.load_state_dict(torch.load('models/phobert_aspect_model.pth', map_location=device))

        # Load sentiment models
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        for aspect in sentiment_aspects:
            model = PhoBERTSentimentClassifier(
                model_name=self.model_name,
                num_classes=4
            ).to(device)
            model.load_state_dict(torch.load(f'models/phobert_sentiment_{aspect}.pth', map_location=device))
            self.sentiment_models[aspect] = model

        print("PhoBERT models loaded successfully!")

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
    print(f"\nInitializing PhoBERT analyzer...")
    analyzer = PhoBERTAspectSentimentAnalyzer()

    # Train models
    print("\n" + "="*50)
    print("PHOBERT ASPECT DETECTION TRAINING")
    print("="*50)

    analyzer.train_aspect_model(train_df, val_df, epochs=30, batch_size=16, lr=2e-5)

    print("\n" + "="*50)
    print("OPTIMIZED PHOBERT SENTIMENT CLASSIFICATION TRAINING")
    print("="*50)

    analyzer.train_sentiment_models(train_df, val_df, epochs=30, batch_size=16, lr=2e-5)

    # Evaluate models
    print("\n" + "="*50)
    print("PHOBERT EVALUATION PHASE")
    print("="*50)

    aspect_results, sentiment_results = analyzer.evaluate_models(test_df)

    # Save results
    results_file = analyzer.save_evaluation_results(aspect_results, sentiment_results)

    print("\n" + "="*50)
    print("PHOBERT TRAINING AND EVALUATION COMPLETED!")
    print("="*50)
    print("Models saved in 'models/' directory:")
    print("  - phobert_aspect_model.pth")
    print("  - phobert_sentiment_*.pth (7 files)")
    print(f"Results saved: {results_file}")

if __name__ == "__main__":
    main()
