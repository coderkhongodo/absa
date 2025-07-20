"""
Evaluation script for PhoBERT Multi-Task Learning model
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
from phobert_multitask import PhoBERTMultiTaskAnalyzer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_phobert_multitask_model():
    """Load trained PhoBERT Multi-Task model"""
    print("Loading PhoBERT Multi-Task model...")
    
    analyzer = PhoBERTMultiTaskAnalyzer()
    analyzer.load_model()
    
    print("PhoBERT Multi-Task model loaded successfully!")
    return analyzer

def evaluate_phobert_multitask_model(analyzer, test_df):
    """Evaluate PhoBERT Multi-Task model with detailed metrics"""
    print("Evaluating PhoBERT Multi-Task model...")
    
    aspect_results, sentiment_results = analyzer.evaluate_multitask_model(test_df)
    
    return aspect_results, sentiment_results

def create_multitask_performance_charts(aspect_results, sentiment_results):
    """Create performance charts for PhoBERT Multi-Task results"""
    os.makedirs('evaluation_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison chart between single-task and multi-task
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PhoBERT Multi-Task vs Single-Task Performance Comparison', fontsize=16)
    
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
    
    # F1-Score comparison
    axes[0, 0].bar(df_aspect['Aspect'], df_aspect['F1-Score'], color='lightblue', alpha=0.7, label='Multi-Task')
    axes[0, 0].set_title('Aspect Detection F1-Score')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    
    # Accuracy comparison
    axes[0, 1].bar(df_aspect['Aspect'], df_aspect['Accuracy'], color='lightgreen', alpha=0.7, label='Multi-Task')
    axes[0, 1].set_title('Aspect Detection Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    
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
    
    # Sentiment Accuracy
    axes[1, 0].bar(df_sentiment['Aspect'], df_sentiment['Accuracy'], color='orange', alpha=0.7, label='Multi-Task')
    axes[1, 0].set_title('Sentiment Classification Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    
    # Sentiment Macro F1
    axes[1, 1].bar(df_sentiment['Aspect'], df_sentiment['Macro F1'], color='pink', alpha=0.7, label='Multi-Task')
    axes[1, 1].set_title('Sentiment Classification Macro F1-Score')
    axes[1, 1].set_ylabel('Macro F1-Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    
    plt.tight_layout()
    chart_file = f'evaluation_results/phobert_multitask_performance_{timestamp}.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed performance heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('PhoBERT Multi-Task Detailed Performance Heatmap', fontsize=16)
    
    # Aspect detection heatmap
    aspect_data = []
    for aspect, metrics in aspect_results.items():
        aspect_data.append([
            metrics['accuracy'],
            metrics['precision'], 
            metrics['recall'],
            metrics['f1_score']
        ])
    
    aspect_df = pd.DataFrame(
        aspect_data,
        index=[aspect.upper() for aspect in aspect_results.keys()],
        columns=['Accuracy', 'Precision', 'Recall', 'F1-Score']
    )
    
    sns.heatmap(aspect_df, annot=True, cmap='Blues', vmin=0, vmax=1, 
                ax=axes[0], cbar_kws={'label': 'Score'})
    axes[0].set_title('Aspect Detection Performance')
    
    # Sentiment classification heatmap
    sentiment_data = []
    for aspect, metrics in sentiment_results.items():
        sentiment_data.append([
            metrics['accuracy'],
            metrics['macro_f1'],
            metrics['weighted_f1']
        ])
    
    sentiment_df = pd.DataFrame(
        sentiment_data,
        index=[aspect.upper() for aspect in sentiment_results.keys()],
        columns=['Accuracy', 'Macro F1', 'Weighted F1']
    )
    
    sns.heatmap(sentiment_df, annot=True, cmap='Oranges', vmin=0, vmax=1,
                ax=axes[1], cbar_kws={'label': 'Score'})
    axes[1].set_title('Sentiment Classification Performance')
    
    plt.tight_layout()
    heatmap_file = f'evaluation_results/phobert_multitask_heatmap_{timestamp}.png'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved:")
    print(f"  - {chart_file}")
    print(f"  - {heatmap_file}")

def save_multitask_results(aspect_results, sentiment_results):
    """Save multi-task evaluation results to JSON file"""
    os.makedirs('evaluation_results', exist_ok=True)
    
    results = {
        'model_type': 'phobert_multitask',
        'model_name': 'vinai/phobert-base-v2',
        'architecture': 'joint_learning',
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
    results_file = f'evaluation_results/phobert_multitask_results_{timestamp}.json'
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    return results_file

def print_multitask_summary(aspect_results, sentiment_results):
    """Print performance summary for multi-task model"""
    print("\n" + "="*70)
    print("PHOBERT MULTI-TASK PERFORMANCE SUMMARY")
    print("="*70)
    
    # Aspect detection summary
    aspect_accuracies = [metrics['accuracy'] for metrics in aspect_results.values()]
    aspect_f1_scores = [metrics['f1_score'] for metrics in aspect_results.values()]
    aspect_precisions = [metrics['precision'] for metrics in aspect_results.values()]
    aspect_recalls = [metrics['recall'] for metrics in aspect_results.values()]
    
    print(f"\nAspect Detection (Multi-Task Learning):")
    print(f"  Average Accuracy:  {np.mean(aspect_accuracies):.4f}")
    print(f"  Average Precision: {np.mean(aspect_precisions):.4f}")
    print(f"  Average Recall:    {np.mean(aspect_recalls):.4f}")
    print(f"  Average F1-Score:  {np.mean(aspect_f1_scores):.4f}")
    
    # Sentiment classification summary
    sentiment_accuracies = [metrics['accuracy'] for metrics in sentiment_results.values()]
    sentiment_macro_f1 = [metrics['macro_f1'] for metrics in sentiment_results.values()]
    sentiment_weighted_f1 = [metrics['weighted_f1'] for metrics in sentiment_results.values()]
    
    print(f"\nSentiment Classification (Multi-Task Learning):")
    print(f"  Average Accuracy:    {np.mean(sentiment_accuracies):.4f}")
    print(f"  Average Macro F1:    {np.mean(sentiment_macro_f1):.4f}")
    print(f"  Average Weighted F1: {np.mean(sentiment_weighted_f1):.4f}")
    
    # Best performing aspects
    best_aspect_f1 = max(aspect_results.items(), key=lambda x: x[1]['f1_score'])
    best_sentiment_acc = max(sentiment_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"\nBest Performing:")
    print(f"  Aspect Detection: {best_aspect_f1[0]} (F1: {best_aspect_f1[1]['f1_score']:.4f})")
    print(f"  Sentiment Classification: {best_sentiment_acc[0]} (Acc: {best_sentiment_acc[1]['accuracy']:.4f})")
    
    # Multi-task advantages
    print(f"\nMulti-Task Learning Advantages:")
    print(f"  ✓ Joint optimization of aspect detection and sentiment classification")
    print(f"  ✓ Shared representations between tasks")
    print(f"  ✓ Reduced overfitting through task regularization")
    print(f"  ✓ More efficient parameter usage")

def main():
    """Main evaluation function for PhoBERT Multi-Task"""
    print("PhoBERT Multi-Task Model Evaluation")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('models/phobert_multitask_model.pth'):
        print("Error: PhoBERT Multi-Task model not found!")
        print("Please run 'python phobert_multitask.py' first to train the model.")
        return
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('data_pr/processed_test.csv')
    print(f"Test data shape: {test_df.shape}")
    
    # Load model
    analyzer = load_phobert_multitask_model()
    
    # Run evaluation
    aspect_results, sentiment_results = evaluate_phobert_multitask_model(analyzer, test_df)
    
    # Print summary
    print_multitask_summary(aspect_results, sentiment_results)
    
    # Save results
    results_file = save_multitask_results(aspect_results, sentiment_results)
    
    # Create charts
    create_multitask_performance_charts(aspect_results, sentiment_results)
    
    print("\n" + "="*70)
    print("PHOBERT MULTI-TASK EVALUATION COMPLETED!")
    print("="*70)
    print(f"Results saved in: evaluation_results/")
    print("Files created:")
    print("  - JSON results file")
    print("  - Performance comparison chart")
    print("  - Detailed performance heatmap")

if __name__ == "__main__":
    main()
