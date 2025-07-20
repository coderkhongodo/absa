"""
Evaluation script for PhoBERT Single-Task Learning model
"""

import pandas as pd
import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
from phobert_single_task import PhoBERTAspectSentimentAnalyzer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_phobert_models():
    """Load trained PhoBERT models"""
    print("Loading PhoBERT models...")
    
    analyzer = PhoBERTAspectSentimentAnalyzer()
    analyzer.load_models()
    
    print("PhoBERT models loaded successfully!")
    return analyzer

def evaluate_phobert_model(analyzer, test_df):
    """Evaluate PhoBERT model with detailed metrics"""
    print("Evaluating PhoBERT model...")
    
    aspect_results, sentiment_results = analyzer.evaluate_models(test_df)
    
    return aspect_results, sentiment_results

def create_performance_charts(aspect_results, sentiment_results):
    """Create performance charts for PhoBERT results"""
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
    fig.suptitle('PhoBERT: Aspect Detection Performance', fontsize=16)
    
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
    aspect_file = f'evaluation_results/phobert_aspect_performance_{timestamp}.png'
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
    fig.suptitle('PhoBERT: Sentiment Classification Performance', fontsize=16)
    
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
    sentiment_file = f'evaluation_results/phobert_sentiment_performance_{timestamp}.png'
    plt.savefig(sentiment_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved:")
    print(f"  - {aspect_file}")
    print(f"  - {sentiment_file}")

def save_results(aspect_results, sentiment_results):
    """Save evaluation results to JSON file"""
    os.makedirs('evaluation_results', exist_ok=True)
    
    results = {
        'model_type': 'phobert_single_task',
        'model_name': 'vinai/phobert-base-v2',
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
    results_file = f'evaluation_results/phobert_results_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    return results_file

def print_summary(aspect_results, sentiment_results):
    """Print performance summary"""
    print("\n" + "="*60)
    print("PHOBERT PERFORMANCE SUMMARY")
    print("="*60)
    
    # Aspect detection summary
    aspect_accuracies = [metrics['accuracy'] for metrics in aspect_results.values()]
    aspect_f1_scores = [metrics['f1_score'] for metrics in aspect_results.values()]
    
    print(f"\nAspect Detection:")
    print(f"  Average Accuracy: {np.mean(aspect_accuracies):.4f}")
    print(f"  Average F1-Score: {np.mean(aspect_f1_scores):.4f}")
    
    # Sentiment classification summary
    sentiment_accuracies = [metrics['accuracy'] for metrics in sentiment_results.values()]
    sentiment_macro_f1 = [metrics['macro_f1'] for metrics in sentiment_results.values()]
    sentiment_weighted_f1 = [metrics['weighted_f1'] for metrics in sentiment_results.values()]
    
    print(f"\nSentiment Classification:")
    print(f"  Average Accuracy: {np.mean(sentiment_accuracies):.4f}")
    print(f"  Average Macro F1: {np.mean(sentiment_macro_f1):.4f}")
    print(f"  Average Weighted F1: {np.mean(sentiment_weighted_f1):.4f}")
    
    # Best performing aspects
    best_aspect_f1 = max(aspect_results.items(), key=lambda x: x[1]['f1_score'])
    best_sentiment_acc = max(sentiment_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\nBest Performing:")
    print(f"  Aspect Detection: {best_aspect_f1[0]} (F1: {best_aspect_f1[1]['f1_score']:.4f})")
    print(f"  Sentiment Classification: {best_sentiment_acc[0]} (Acc: {best_sentiment_acc[1]['accuracy']:.4f})")

def main():
    """Main evaluation function"""
    print("PhoBERT Single-Task Model Evaluation")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists('models/phobert_aspect_model.pth'):
        print("Error: PhoBERT models not found!")
        print("Please run 'python phobert_single_task.py' first to train the models.")
        return
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data_pr/processed_test.csv')
    print(f"Test data shape: {test_df.shape}")
    
    # Load model
    analyzer = load_phobert_models()
    
    # Run evaluation
    aspect_results, sentiment_results = evaluate_phobert_model(analyzer, test_df)
    
    # Print summary
    print_summary(aspect_results, sentiment_results)
    
    # Save results
    results_file = save_results(aspect_results, sentiment_results)
    
    # Create charts
    create_performance_charts(aspect_results, sentiment_results)
    
    print("\n" + "="*60)
    print("PHOBERT EVALUATION COMPLETED!")
    print("="*60)
    print(f"Results saved in: evaluation_results/")
    print("Files created:")
    print("  - JSON results file")
    print("  - Aspect performance chart")
    print("  - Sentiment performance chart")

if __name__ == "__main__":
    main()
