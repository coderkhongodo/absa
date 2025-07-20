"""
Evaluation script for Multi-Task Learning model
"""

import pandas as pd
import numpy as np
import torch
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
from multi_task_learning_clean import MultiTaskAspectSentimentAnalyzer, MultiTaskBiLSTM, MultiTaskDataset, SimpleTokenizer
from torch.utils.data import DataLoader

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_multitask_model():
    """Load trained multi-task model and tokenizer"""
    print("Loading multi-task model...")
    
    # Initialize analyzer
    analyzer = MultiTaskAspectSentimentAnalyzer()
    
    # Load tokenizer
    with open('models/multi_task_tokenizer.pkl', 'rb') as f:
        analyzer.tokenizer = pickle.load(f)
    
    # Load model
    analyzer.model = MultiTaskBiLSTM(
        vocab_size=analyzer.tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=200,
        num_aspects=8
    ).to(device)
    analyzer.model.load_state_dict(torch.load('models/multi_task_model.pth', map_location=device))
    
    print(f"Loaded multi-task model with vocab size: {analyzer.tokenizer.vocab_size}")
    return analyzer

def evaluate_multitask_model(analyzer, test_df):
    """Evaluate multi-task model with detailed metrics"""
    print("Evaluating multi-task model...")
    
    # Prepare test data
    test_texts = test_df['processed_data'].tolist()
    test_aspect_labels = analyzer.prepare_aspect_labels(test_df)
    test_sentiment_labels = analyzer.prepare_sentiment_labels(test_df)
    
    test_dataset = MultiTaskDataset(test_texts, test_aspect_labels, 
                                  test_sentiment_labels, analyzer.tokenizer, analyzer.max_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    analyzer.model.eval()
    
    # Collect predictions
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = {aspect: [] for aspect in ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']}
    all_sentiment_labels = {aspect: [] for aspect in ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']}
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx+1}/{len(test_loader)}")
                
            input_ids = batch['input_ids'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            
            # Forward pass
            aspect_probs, sentiment_outputs = analyzer.model(input_ids)
            
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
    print("\n" + "="*60)
    print("ASPECT DETECTION EVALUATION")
    print("="*60)
    
    aspect_results = {}
    for i, aspect in enumerate(analyzer.aspect_names):
        y_true = all_aspect_labels[:, i]
        y_pred = all_aspect_preds[:, i]
        
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate precision, recall, f1
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle case where only one class is present
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
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\n{aspect.upper()}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
    
    # Evaluate sentiment classification
    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION EVALUATION")
    print("="*60)
    
    sentiment_results = {}
    for aspect in all_sentiment_preds.keys():
        print(f"\n{aspect.upper()} SENTIMENT:")
        
        y_true = np.array(all_sentiment_labels[aspect])
        y_pred = np.array(all_sentiment_preds[aspect])
        
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        
        # Get classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=['No Aspect', 'Negative', 'Neutral', 'Positive'],
                                     labels=[0, 1, 2, 3],
                                     output_dict=True, zero_division=0)
        
        sentiment_results[aspect] = {
            'accuracy': accuracy,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
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

def save_results(aspect_results, sentiment_results):
    """Save evaluation results to JSON file"""
    os.makedirs('evaluation_results', exist_ok=True)
    
    results = {
        'model_type': 'multi_task',
        'aspect_detection': aspect_results,
        'sentiment_classification': sentiment_results,
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert numpy types to Python types for JSON serialization
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'evaluation_results/multitask_results_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    return results_file

def create_comparison_charts(aspect_results, sentiment_results):
    """Create comparison charts for multi-task results"""
    os.makedirs('evaluation_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Aspect detection performance
    aspect_metrics = []
    for aspect, metrics in aspect_results.items():
        aspect_metrics.append({
            'Aspect': aspect.upper(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
    
    df_aspect = pd.DataFrame(aspect_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Task: Aspect Detection Performance', fontsize=16)
    
    # Precision
    axes[0, 0].bar(df_aspect['Aspect'], df_aspect['Precision'], color='skyblue')
    axes[0, 0].set_title('Precision by Aspect')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Recall
    axes[0, 1].bar(df_aspect['Aspect'], df_aspect['Recall'], color='lightgreen')
    axes[0, 1].set_title('Recall by Aspect')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    
    # F1-Score
    axes[1, 0].bar(df_aspect['Aspect'], df_aspect['F1-Score'], color='orange')
    axes[1, 0].set_title('F1-Score by Aspect')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    
    # Accuracy
    axes[1, 1].bar(df_aspect['Aspect'], df_aspect['Accuracy'], color='pink')
    axes[1, 1].set_title('Accuracy by Aspect')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    aspect_file = f'evaluation_results/multitask_aspect_performance_{timestamp}.png'
    plt.savefig(aspect_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sentiment classification performance
    sentiment_metrics = []
    for aspect, metrics in sentiment_results.items():
        sentiment_metrics.append({
            'Aspect': aspect.upper(),
            'Accuracy': metrics['accuracy'],
            'Macro F1': metrics['macro_f1'],
            'Weighted F1': metrics['weighted_f1']
        })
    
    df_sentiment = pd.DataFrame(sentiment_metrics)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Multi-Task: Sentiment Classification Performance', fontsize=16)
    
    # Accuracy
    axes[0].bar(df_sentiment['Aspect'], df_sentiment['Accuracy'], color='lightblue')
    axes[0].set_title('Accuracy by Aspect')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0, 1)
    
    # Macro F1
    axes[1].bar(df_sentiment['Aspect'], df_sentiment['Macro F1'], color='lightcoral')
    axes[1].set_title('Macro F1-Score by Aspect')
    axes[1].set_ylabel('Macro F1-Score')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1)
    
    # Weighted F1
    axes[2].bar(df_sentiment['Aspect'], df_sentiment['Weighted F1'], color='lightyellow')
    axes[2].set_title('Weighted F1-Score by Aspect')
    axes[2].set_ylabel('Weighted F1-Score')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    sentiment_file = f'evaluation_results/multitask_sentiment_performance_{timestamp}.png'
    plt.savefig(sentiment_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved:")
    print(f"  - {aspect_file}")
    print(f"  - {sentiment_file}")

def main():
    """Main evaluation function"""
    print("Multi-Task Model Evaluation")
    print("="*60)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data_pr/processed_test.csv')
    print(f"Test data shape: {test_df.shape}")
    
    # Load model
    analyzer = load_multitask_model()
    
    # Run evaluation
    aspect_results, sentiment_results = evaluate_multitask_model(analyzer, test_df)
    
    # Save results
    results_file = save_results(aspect_results, sentiment_results)
    
    # Create charts
    create_comparison_charts(aspect_results, sentiment_results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)
    print(f"Results saved in: evaluation_results/")
    print("Files created:")
    print("  - JSON results file")
    print("  - Aspect performance chart")
    print("  - Sentiment performance chart")

if __name__ == "__main__":
    main()
