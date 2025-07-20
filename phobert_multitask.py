"""
Optimized PhoBERT Multi-Task Learning for Vietnamese Aspect-based Sentiment Analysis
Joint training with enhanced convergence optimization:
- Cosine Annealing with Warm Restarts (T_0=5, T_mult=2)
- Label Smoothing for both aspect and sentiment tasks
- Gradient Clipping (max_norm=1.0) for stability
- Combined F1-based early stopping (aspect + sentiment)
- Enhanced learning rate (2e-5) with real-time monitoring
- Multi-task loss weighting and regularization
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
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
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class PhoBERTMultiTaskDataset(Dataset):
    """Multi-task dataset for PhoBERT with aspect and sentiment labels"""
    def __init__(self, texts: List[str], aspect_labels: np.ndarray, sentiment_labels: Dict[str, np.ndarray], 
                 tokenizer, max_length: int = 128):
        self.texts = texts
        self.aspect_labels = aspect_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts for faster training
        print("Pre-tokenizing texts for multi-task learning...")
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
        # Prepare sentiment labels for all aspects
        sentiment_dict = {}
        for aspect, labels in self.sentiment_labels.items():
            sentiment_dict[aspect] = torch.tensor(labels[idx], dtype=torch.long)
        
        return {
            'input_ids': self.tokenized_texts[idx]['input_ids'],
            'attention_mask': self.tokenized_texts[idx]['attention_mask'],
            'aspect_labels': torch.tensor(self.aspect_labels[idx], dtype=torch.float),
            'sentiment_labels': sentiment_dict
        }
    
    def preprocess_vietnamese_text(self, text) -> str:
        """Preprocess Vietnamese text for PhoBERT"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', ' ', text)
        return text.strip()

class PhoBERTMultiTaskModel(nn.Module):
    """PhoBERT Multi-Task Model for joint aspect detection and sentiment classification"""
    def __init__(self, model_name: str = "vinai/phobert-base-v2", num_aspects: int = 8, 
                 num_sentiment_classes: int = 4, dropout: float = 0.1, freeze_layers: int = 4):
        super(PhoBERTMultiTaskModel, self).__init__()
        
        # Shared PhoBERT encoder
        self.phobert = AutoModel.from_pretrained(model_name)
        
        # Freeze fewer layers for multi-task learning
        if freeze_layers > 0:
            for param in self.phobert.embeddings.parameters():
                param.requires_grad = False
            
            for i in range(freeze_layers):
                if i < len(self.phobert.encoder.layer):
                    for param in self.phobert.encoder.layer[i].parameters():
                        param.requires_grad = False
        
        hidden_size = self.phobert.config.hidden_size
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512)
        )
        
        # Aspect detection head
        self.aspect_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_aspects)
        )
        
        # Sentiment classification heads for each aspect
        self.sentiment_heads = nn.ModuleDict()
        sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        
        for aspect in sentiment_aspects:
            self.sentiment_heads[aspect] = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_sentiment_classes)
            )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask):
        # Get PhoBERT outputs
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation with layer norm
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        # Shared representation
        shared_repr = self.shared_layer(pooled_output)
        
        # Aspect detection
        aspect_logits = self.aspect_head(shared_repr)
        
        # Sentiment classification for each aspect
        sentiment_logits = {}
        for aspect, head in self.sentiment_heads.items():
            sentiment_logits[aspect] = head(shared_repr)
        
        return aspect_logits, sentiment_logits

class PhoBERTMultiTaskAnalyzer:
    """PhoBERT Multi-Task Analyzer for joint aspect and sentiment learning"""
    
    def __init__(self, model_name: str = "vinai/phobert-base-v2", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Model
        self.model = None
        
        # Aspect names
        self.aspect_names = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower', 'others']
        self.sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
        
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
        """Prepare sentiment labels for all aspects"""
        sentiment_labels = {}
        
        for aspect in self.sentiment_aspects:
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
        
        return sentiment_labels
    
    def train_multitask_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                             epochs: int = 30, batch_size: int = 16, lr: float = 1e-5,
                             aspect_weight: float = 1.0, sentiment_weight: float = 1.0):
        """Train multi-task model with joint optimization"""
        print("Training PhoBERT Multi-Task model...")
        print(f"Training on full dataset: {len(train_df)} samples")
        
        # Prepare data
        train_texts = train_df['processed_data'].tolist()
        val_texts = val_df['processed_data'].tolist()
        
        train_aspect_labels = self.prepare_aspect_labels(train_df)
        val_aspect_labels = self.prepare_aspect_labels(val_df)
        
        train_sentiment_labels = self.prepare_sentiment_labels(train_df)
        val_sentiment_labels = self.prepare_sentiment_labels(val_df)
        
        # Create datasets
        train_dataset = PhoBERTMultiTaskDataset(
            train_texts, train_aspect_labels, train_sentiment_labels, 
            self.tokenizer, self.max_length
        )
        val_dataset = PhoBERTMultiTaskDataset(
            val_texts, val_aspect_labels, val_sentiment_labels, 
            self.tokenizer, self.max_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
        
        # Initialize model
        self.model = PhoBERTMultiTaskModel(
            model_name=self.model_name,
            num_aspects=len(self.aspect_names),
            num_sentiment_classes=4,
            dropout=0.1,
            freeze_layers=4
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

        class LabelSmoothingCrossEntropyLoss(nn.Module):
            def __init__(self, smoothing=0.1, num_classes=4):
                super().__init__()
                self.smoothing = smoothing
                self.num_classes = num_classes
                self.ce = nn.CrossEntropyLoss()

            def forward(self, pred, target):
                # Convert to one-hot and apply smoothing
                confidence = 1.0 - self.smoothing
                smooth_value = self.smoothing / (self.num_classes - 1)

                # Create smoothed labels
                one_hot = torch.zeros_like(pred)
                one_hot.scatter_(1, target.unsqueeze(1), confidence)
                one_hot += smooth_value
                one_hot.scatter_(1, target.unsqueeze(1), confidence)

                # Use KL divergence for smoothed labels
                log_probs = F.log_softmax(pred, dim=1)
                return -torch.sum(one_hot * log_probs, dim=1).mean()

        # Optimized loss functions with label smoothing
        aspect_criterion = LabelSmoothingBCELoss(smoothing=0.1)
        sentiment_criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1, num_classes=4)
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(), 
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
        
        # Enhanced training loop with multi-task early stopping
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        patience = 7
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_aspect_loss = 0
            train_sentiment_loss = 0
            train_total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                aspect_labels = batch['aspect_labels'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                aspect_logits, sentiment_logits = self.model(input_ids, attention_mask)
                
                # Aspect loss
                aspect_loss = aspect_criterion(aspect_logits, aspect_labels)
                
                # Sentiment losses
                sentiment_loss = 0
                for aspect in self.sentiment_aspects:
                    sentiment_labels = batch['sentiment_labels'][aspect].to(device)
                    sentiment_loss += sentiment_criterion(sentiment_logits[aspect], sentiment_labels)
                
                sentiment_loss /= len(self.sentiment_aspects)  # Average sentiment loss
                
                # Combined loss
                total_loss = aspect_weight * aspect_loss + sentiment_weight * sentiment_loss
                
                total_loss.backward()

                # Gradient clipping for stability and faster convergence
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()  # Step scheduler after each batch for cosine annealing

                train_aspect_loss += aspect_loss.item()
                train_sentiment_loss += sentiment_loss.item()
                train_total_loss += total_loss.item()

                # Update progress bar with current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'aspect_loss': train_aspect_loss / (len(progress_bar.iterable) if hasattr(progress_bar, 'iterable') else 1),
                    'sentiment_loss': train_sentiment_loss / (len(progress_bar.iterable) if hasattr(progress_bar, 'iterable') else 1),
                    'total_loss': train_total_loss / (len(progress_bar.iterable) if hasattr(progress_bar, 'iterable') else 1),
                    'lr': f'{current_lr:.2e}'
                })
            
            # Validation
            self.model.eval()
            val_aspect_loss = 0
            val_sentiment_loss = 0
            val_total_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    aspect_labels = batch['aspect_labels'].to(device)
                    
                    # Forward pass
                    aspect_logits, sentiment_logits = self.model(input_ids, attention_mask)
                    
                    # Aspect loss
                    aspect_loss = aspect_criterion(aspect_logits, aspect_labels)
                    
                    # Sentiment losses
                    sentiment_loss = 0
                    for aspect in self.sentiment_aspects:
                        sentiment_labels = batch['sentiment_labels'][aspect].to(device)
                        sentiment_loss += sentiment_criterion(sentiment_logits[aspect], sentiment_labels)
                    
                    sentiment_loss /= len(self.sentiment_aspects)
                    
                    # Combined loss
                    total_loss = aspect_weight * aspect_loss + sentiment_weight * sentiment_loss
                    
                    val_aspect_loss += aspect_loss.item()
                    val_sentiment_loss += sentiment_loss.item()
                    val_total_loss += total_loss.item()
            
            # Average losses
            train_aspect_loss /= len(train_loader)
            train_sentiment_loss /= len(train_loader)
            train_total_loss /= len(train_loader)
            
            val_aspect_loss /= len(val_loader)
            val_sentiment_loss /= len(val_loader)
            val_total_loss /= len(val_loader)

            # Quick validation F1 calculation for multi-task early stopping
            self.model.eval()
            val_aspect_f1_scores = []
            val_sentiment_f1_scores = []

            with torch.no_grad():
                val_aspect_preds = []
                val_aspect_labels = []
                val_sentiment_preds = {aspect: [] for aspect in self.sentiment_aspects}
                val_sentiment_labels = {aspect: [] for aspect in self.sentiment_aspects}

                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    aspect_labels = batch['aspect_labels'].to(device)

                    # Forward pass
                    aspect_logits, sentiment_logits = self.model(input_ids, attention_mask)

                    # Aspect predictions
                    aspect_preds = torch.sigmoid(aspect_logits) > 0.5
                    val_aspect_preds.extend(aspect_preds.cpu().numpy())
                    val_aspect_labels.extend(aspect_labels.cpu().numpy())

                    # Sentiment predictions
                    for aspect in self.sentiment_aspects:
                        sentiment_labels = batch['sentiment_labels'][aspect].to(device)
                        sentiment_preds = torch.argmax(sentiment_logits[aspect], dim=1)

                        val_sentiment_preds[aspect].extend(sentiment_preds.cpu().numpy())
                        val_sentiment_labels[aspect].extend(sentiment_labels.cpu().numpy())

                # Calculate aspect F1 scores
                val_aspect_preds = np.array(val_aspect_preds)
                val_aspect_labels = np.array(val_aspect_labels)

                for i in range(len(self.aspect_names)):
                    y_true = val_aspect_labels[:, i]
                    y_pred = val_aspect_preds[:, i]

                    # Calculate F1
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    val_aspect_f1_scores.append(f1)

                # Calculate sentiment F1 scores
                for aspect in self.sentiment_aspects:
                    y_true = np.array(val_sentiment_labels[aspect])
                    y_pred = np.array(val_sentiment_preds[aspect])

                    # Calculate macro F1 for sentiment
                    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                    val_sentiment_f1_scores.append(f1)

                # Combined F1 score (weighted average)
                avg_aspect_f1 = np.mean(val_aspect_f1_scores)
                avg_sentiment_f1 = np.mean(val_sentiment_f1_scores)
                combined_f1 = 0.5 * avg_aspect_f1 + 0.5 * avg_sentiment_f1  # Equal weight

            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Aspect: {train_aspect_loss:.4f}, Sentiment: {train_sentiment_loss:.4f}, Total: {train_total_loss:.4f}")
            print(f"  Val   - Aspect: {val_aspect_loss:.4f}, Sentiment: {val_sentiment_loss:.4f}, Total: {val_total_loss:.4f}")
            print(f"  Val F1 - Aspect: {avg_aspect_f1:.4f}, Sentiment: {avg_sentiment_f1:.4f}, Combined: {combined_f1:.4f}")

            # Enhanced early stopping based on combined F1 score
            if combined_f1 > best_val_f1 + 0.001:  # F1 improvement threshold
                best_val_f1 = combined_f1
                best_val_loss = val_total_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/phobert_multitask_model.pth')
                print(f"  → Model improved (Combined F1: {combined_f1:.4f}), saved checkpoint")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/phobert_multitask_model.pth'))
        print("PhoBERT Multi-Task model training completed!")

    def evaluate_multitask_model(self, test_df: pd.DataFrame):
        """Evaluate multi-task model on test data"""
        print("Evaluating PhoBERT Multi-Task model...")

        test_texts = test_df['processed_data'].tolist()
        test_aspect_labels = self.prepare_aspect_labels(test_df)
        test_sentiment_labels = self.prepare_sentiment_labels(test_df)

        test_dataset = PhoBERTMultiTaskDataset(
            test_texts, test_aspect_labels, test_sentiment_labels,
            self.tokenizer, self.max_length
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()

        # Collect predictions
        all_aspect_preds = []
        all_aspect_labels = []
        all_sentiment_preds = {aspect: [] for aspect in self.sentiment_aspects}
        all_sentiment_labels = {aspect: [] for aspect in self.sentiment_aspects}

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                aspect_labels = batch['aspect_labels'].to(device)

                # Forward pass
                aspect_logits, sentiment_logits = self.model(input_ids, attention_mask)

                # Aspect predictions
                aspect_preds = torch.sigmoid(aspect_logits) > 0.5
                all_aspect_preds.extend(aspect_preds.cpu().numpy())
                all_aspect_labels.extend(aspect_labels.cpu().numpy())

                # Sentiment predictions
                for aspect in self.sentiment_aspects:
                    sentiment_labels = batch['sentiment_labels'][aspect].to(device)
                    sentiment_preds = torch.argmax(sentiment_logits[aspect], dim=1)

                    all_sentiment_preds[aspect].extend(sentiment_preds.cpu().numpy())
                    all_sentiment_labels[aspect].extend(sentiment_labels.cpu().numpy())

        # Evaluate aspect detection
        print("\n" + "="*50)
        print("ASPECT DETECTION RESULTS")
        print("="*50)

        all_aspect_preds = np.array(all_aspect_preds)
        all_aspect_labels = np.array(all_aspect_labels)

        aspect_results = {}
        for i, aspect in enumerate(self.aspect_names):
            y_true = all_aspect_labels[:, i]
            y_pred = all_aspect_preds[:, i]

            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0

            aspect_results[aspect] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            print(f"{aspect.upper()}: Acc={accuracy:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        # Evaluate sentiment classification
        print("\n" + "="*50)
        print("SENTIMENT CLASSIFICATION RESULTS")
        print("="*50)

        sentiment_results = {}
        for aspect in self.sentiment_aspects:
            y_true = np.array(all_sentiment_labels[aspect])
            y_pred = np.array(all_sentiment_preds[aspect])

            accuracy = accuracy_score(y_true, y_pred)

            # Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=['No Aspect', 'Negative', 'Neutral', 'Positive'],
                labels=[0, 1, 2, 3],
                output_dict=True,
                zero_division=0
            )

            sentiment_results[aspect] = {
                'accuracy': accuracy,
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'classification_report': report
            }

            print(f"{aspect.upper()}: Acc={accuracy:.3f}, Macro F1={report['macro avg']['f1-score']:.3f}, Weighted F1={report['weighted avg']['f1-score']:.3f}")

        return aspect_results, sentiment_results

    def save_results(self, aspect_results: Dict, sentiment_results: Dict):
        """Save evaluation results"""
        os.makedirs('evaluation_results', exist_ok=True)

        results = {
            'model_type': 'phobert_multitask',
            'model_name': self.model_name,
            'max_length': self.max_length,
            'aspect_detection': aspect_results,
            'sentiment_classification': sentiment_results,
            'timestamp': datetime.now().isoformat()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'evaluation_results/phobert_multitask_results_{timestamp}.json'

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nResults saved to: {results_file}")
        return results_file

    def load_model(self):
        """Load trained multi-task model"""
        print("Loading PhoBERT Multi-Task model...")

        self.model = PhoBERTMultiTaskModel(
            model_name=self.model_name,
            num_aspects=len(self.aspect_names),
            num_sentiment_classes=4,
            dropout=0.1,
            freeze_layers=4
        ).to(device)

        self.model.load_state_dict(torch.load('models/phobert_multitask_model.pth', map_location=device))
        print("PhoBERT Multi-Task model loaded successfully!")

def main():
    """Main function for PhoBERT Multi-Task Learning"""
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
    print(f"\nInitializing PhoBERT Multi-Task analyzer...")
    analyzer = PhoBERTMultiTaskAnalyzer(max_length=128)

    # Train multi-task model
    print("\n" + "="*60)
    print("PHOBERT MULTI-TASK TRAINING")
    print("="*60)

    analyzer.train_multitask_model(
        train_df, val_df,
        epochs=30,
        batch_size=16,
        lr=2e-5,               # Increased learning rate for faster convergence
        aspect_weight=1.0,      # Weight for aspect detection loss
        sentiment_weight=1.0    # Weight for sentiment classification loss
    )

    # Evaluate model
    print("\n" + "="*60)
    print("PHOBERT MULTI-TASK EVALUATION")
    print("="*60)

    aspect_results, sentiment_results = analyzer.evaluate_multitask_model(test_df)

    # Save results
    results_file = analyzer.save_results(aspect_results, sentiment_results)

    # Print summary
    print("\n" + "="*60)
    print("PHOBERT MULTI-TASK TRAINING COMPLETED!")
    print("="*60)

    # Calculate averages
    avg_aspect_acc = np.mean([r['accuracy'] for r in aspect_results.values()])
    avg_aspect_f1 = np.mean([r['f1_score'] for r in aspect_results.values()])
    avg_sentiment_acc = np.mean([r['accuracy'] for r in sentiment_results.values()])
    avg_sentiment_f1 = np.mean([r['macro_f1'] for r in sentiment_results.values()])

    print(f"Average Aspect Detection - Accuracy: {avg_aspect_acc:.3f}, F1: {avg_aspect_f1:.3f}")
    print(f"Average Sentiment Classification - Accuracy: {avg_sentiment_acc:.3f}, Macro F1: {avg_sentiment_f1:.3f}")

    print("\nModel saved:")
    print("  - phobert_multitask_model.pth")
    print(f"Results: {results_file}")

if __name__ == "__main__":
    main()
