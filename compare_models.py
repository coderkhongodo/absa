"""
Model Comparison Script for Vietnamese Aspect-based Sentiment Analysis
Compare performance between BiLSTM Single-Task, BiLSTM Multi-Task, 
PhoBERT Single-Task, and PhoBERT Multi-Task models
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

def load_all_results():
    """Load all available model results"""
    results = {}
    
    # Find all result files
    result_files = glob.glob('evaluation_results/*_results_*.json')
    
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            model_type = data.get('model_type', 'unknown')
            
            # Extract model info
            if 'bilstm_single' in model_type:
                model_name = 'BiLSTM Single-Task'
            elif 'bilstm_multi' in model_type:
                model_name = 'BiLSTM Multi-Task'
            elif 'phobert_single' in model_type:
                model_name = 'PhoBERT Single-Task'
            elif 'phobert_multitask' in model_type:
                model_name = 'PhoBERT Multi-Task'
            else:
                model_name = model_type
            
            results[model_name] = data
            print(f"Loaded results for: {model_name}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def create_comparison_charts(results):
    """Create comprehensive comparison charts"""
    if not results:
        print("No results found to compare!")
        return
    
    os.makedirs('evaluation_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for comparison
    models = list(results.keys())
    aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower', 'others']
    sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
    
    # 1. Aspect Detection Comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model Performance Comparison - Aspect Detection', fontsize=20)
    
    # Accuracy comparison
    aspect_acc_data = []
    for model in models:
        if 'aspect_detection' in results[model]:
            for aspect in aspects:
                if aspect in results[model]['aspect_detection']:
                    aspect_acc_data.append({
                        'Model': model,
                        'Aspect': aspect.upper(),
                        'Accuracy': results[model]['aspect_detection'][aspect]['accuracy']
                    })
    
    if aspect_acc_data:
        df_aspect_acc = pd.DataFrame(aspect_acc_data)
        pivot_acc = df_aspect_acc.pivot(index='Aspect', columns='Model', values='Accuracy')
        pivot_acc.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Aspect Detection Accuracy by Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    aspect_f1_data = []
    for model in models:
        if 'aspect_detection' in results[model]:
            for aspect in aspects:
                if aspect in results[model]['aspect_detection']:
                    aspect_f1_data.append({
                        'Model': model,
                        'Aspect': aspect.upper(),
                        'F1-Score': results[model]['aspect_detection'][aspect]['f1_score']
                    })
    
    if aspect_f1_data:
        df_aspect_f1 = pd.DataFrame(aspect_f1_data)
        pivot_f1 = df_aspect_f1.pivot(index='Aspect', columns='Model', values='F1-Score')
        pivot_f1.plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Aspect Detection F1-Score by Model')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Average performance comparison
    avg_metrics = []
    for model in models:
        if 'aspect_detection' in results[model]:
            accuracies = [results[model]['aspect_detection'][aspect]['accuracy'] 
                         for aspect in aspects if aspect in results[model]['aspect_detection']]
            f1_scores = [results[model]['aspect_detection'][aspect]['f1_score'] 
                        for aspect in aspects if aspect in results[model]['aspect_detection']]
            
            if accuracies and f1_scores:
                avg_metrics.append({
                    'Model': model,
                    'Avg Accuracy': np.mean(accuracies),
                    'Avg F1-Score': np.mean(f1_scores)
                })
    
    if avg_metrics:
        df_avg = pd.DataFrame(avg_metrics)
        x_pos = np.arange(len(df_avg))
        
        axes[1, 0].bar(x_pos - 0.2, df_avg['Avg Accuracy'], 0.4, label='Accuracy', alpha=0.8)
        axes[1, 0].bar(x_pos + 0.2, df_avg['Avg F1-Score'], 0.4, label='F1-Score', alpha=0.8)
        axes[1, 0].set_title('Average Aspect Detection Performance')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(df_avg['Model'], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
    
    # Model architecture comparison
    model_info = []
    for model in models:
        if 'BiLSTM' in model:
            architecture = 'BiLSTM'
            task_type = 'Single-Task' if 'Single' in model else 'Multi-Task'
        else:
            architecture = 'PhoBERT'
            task_type = 'Single-Task' if 'Single' in model else 'Multi-Task'
        
        model_info.append({
            'Model': model,
            'Architecture': architecture,
            'Task Type': task_type
        })
    
    # Create architecture comparison
    if model_info:
        df_info = pd.DataFrame(model_info)
        arch_counts = df_info.groupby(['Architecture', 'Task Type']).size().unstack(fill_value=0)
        arch_counts.plot(kind='bar', ax=axes[1, 1], width=0.6)
        axes[1, 1].set_title('Model Architecture Distribution')
        axes[1, 1].set_ylabel('Number of Models')
        axes[1, 1].legend(title='Task Type')
        axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    aspect_file = f'evaluation_results/model_comparison_aspects_{timestamp}.png'
    plt.savefig(aspect_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sentiment Classification Comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Model Performance Comparison - Sentiment Classification', fontsize=20)
    
    # Sentiment accuracy comparison
    sentiment_acc_data = []
    for model in models:
        if 'sentiment_classification' in results[model]:
            for aspect in sentiment_aspects:
                if aspect in results[model]['sentiment_classification']:
                    sentiment_acc_data.append({
                        'Model': model,
                        'Aspect': aspect.upper(),
                        'Accuracy': results[model]['sentiment_classification'][aspect]['accuracy']
                    })
    
    if sentiment_acc_data:
        df_sentiment_acc = pd.DataFrame(sentiment_acc_data)
        pivot_sent_acc = df_sentiment_acc.pivot(index='Aspect', columns='Model', values='Accuracy')
        pivot_sent_acc.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Sentiment Classification Accuracy by Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Sentiment macro F1 comparison
    sentiment_f1_data = []
    for model in models:
        if 'sentiment_classification' in results[model]:
            for aspect in sentiment_aspects:
                if aspect in results[model]['sentiment_classification']:
                    sentiment_f1_data.append({
                        'Model': model,
                        'Aspect': aspect.upper(),
                        'Macro F1': results[model]['sentiment_classification'][aspect]['macro_f1']
                    })
    
    if sentiment_f1_data:
        df_sentiment_f1 = pd.DataFrame(sentiment_f1_data)
        pivot_sent_f1 = df_sentiment_f1.pivot(index='Aspect', columns='Model', values='Macro F1')
        pivot_sent_f1.plot(kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Sentiment Classification Macro F1-Score by Model')
        axes[0, 1].set_ylabel('Macro F1-Score')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Overall performance heatmap
    performance_matrix = []
    metrics = ['Aspect Accuracy', 'Aspect F1', 'Sentiment Accuracy', 'Sentiment Macro F1']
    
    for model in models:
        row = []
        
        # Aspect metrics
        if 'aspect_detection' in results[model]:
            aspect_accs = [results[model]['aspect_detection'][aspect]['accuracy'] 
                          for aspect in aspects if aspect in results[model]['aspect_detection']]
            aspect_f1s = [results[model]['aspect_detection'][aspect]['f1_score'] 
                         for aspect in aspects if aspect in results[model]['aspect_detection']]
            row.extend([np.mean(aspect_accs) if aspect_accs else 0, 
                       np.mean(aspect_f1s) if aspect_f1s else 0])
        else:
            row.extend([0, 0])
        
        # Sentiment metrics
        if 'sentiment_classification' in results[model]:
            sent_accs = [results[model]['sentiment_classification'][aspect]['accuracy'] 
                        for aspect in sentiment_aspects if aspect in results[model]['sentiment_classification']]
            sent_f1s = [results[model]['sentiment_classification'][aspect]['macro_f1'] 
                       for aspect in sentiment_aspects if aspect in results[model]['sentiment_classification']]
            row.extend([np.mean(sent_accs) if sent_accs else 0, 
                       np.mean(sent_f1s) if sent_f1s else 0])
        else:
            row.extend([0, 0])
        
        performance_matrix.append(row)
    
    if performance_matrix:
        df_heatmap = pd.DataFrame(performance_matrix, index=models, columns=metrics)
        sns.heatmap(df_heatmap, annot=True, cmap='RdYlBu_r', vmin=0, vmax=1, 
                   ax=axes[1, 0], cbar_kws={'label': 'Score'})
        axes[1, 0].set_title('Overall Performance Heatmap')
    
    # Best model per metric
    if performance_matrix:
        best_models = []
        for i, metric in enumerate(metrics):
            scores = [row[i] for row in performance_matrix]
            best_idx = np.argmax(scores)
            best_models.append({
                'Metric': metric,
                'Best Model': models[best_idx],
                'Score': scores[best_idx]
            })
        
        df_best = pd.DataFrame(best_models)
        colors = ['gold', 'silver', 'lightblue', 'lightgreen']
        bars = axes[1, 1].bar(df_best['Metric'], df_best['Score'], color=colors)
        axes[1, 1].set_title('Best Model Performance by Metric')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(0, 1)
        
        # Add model names on bars
        for i, (bar, model) in enumerate(zip(bars, df_best['Best Model'])):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           model, ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    sentiment_file = f'evaluation_results/model_comparison_sentiment_{timestamp}.png'
    plt.savefig(sentiment_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison charts saved:")
    print(f"  - {aspect_file}")
    print(f"  - {sentiment_file}")

def print_comparison_summary(results):
    """Print detailed comparison summary"""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    if not results:
        print("No results available for comparison!")
        return
    
    aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower', 'others']
    sentiment_aspects = ['smell', 'texture', 'colour', 'price', 'shipping', 'packing', 'stayingpower']
    
    # Overall rankings
    model_scores = {}
    
    for model_name, data in results.items():
        scores = []
        
        # Aspect detection scores
        if 'aspect_detection' in data:
            aspect_f1s = [data['aspect_detection'][aspect]['f1_score'] 
                         for aspect in aspects if aspect in data['aspect_detection']]
            if aspect_f1s:
                scores.append(np.mean(aspect_f1s))
        
        # Sentiment classification scores
        if 'sentiment_classification' in data:
            sent_f1s = [data['sentiment_classification'][aspect]['macro_f1'] 
                       for aspect in sentiment_aspects if aspect in data['sentiment_classification']]
            if sent_f1s:
                scores.append(np.mean(sent_f1s))
        
        model_scores[model_name] = np.mean(scores) if scores else 0
    
    # Rank models
    ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nOVERALL MODEL RANKING (by average F1-score):")
    print("-" * 50)
    for i, (model, score) in enumerate(ranked_models, 1):
        print(f"{i}. {model}: {score:.4f}")
    
    # Detailed breakdown
    print(f"\nDETAILED PERFORMANCE BREAKDOWN:")
    print("-" * 50)
    
    for model_name, data in results.items():
        print(f"\n{model_name.upper()}:")
        
        # Aspect detection
        if 'aspect_detection' in data:
            aspect_accs = [data['aspect_detection'][aspect]['accuracy'] 
                          for aspect in aspects if aspect in data['aspect_detection']]
            aspect_f1s = [data['aspect_detection'][aspect]['f1_score'] 
                         for aspect in aspects if aspect in data['aspect_detection']]
            
            if aspect_accs and aspect_f1s:
                print(f"  Aspect Detection - Avg Accuracy: {np.mean(aspect_accs):.4f}, Avg F1: {np.mean(aspect_f1s):.4f}")
        
        # Sentiment classification
        if 'sentiment_classification' in data:
            sent_accs = [data['sentiment_classification'][aspect]['accuracy'] 
                        for aspect in sentiment_aspects if aspect in data['sentiment_classification']]
            sent_f1s = [data['sentiment_classification'][aspect]['macro_f1'] 
                       for aspect in sentiment_aspects if aspect in data['sentiment_classification']]
            
            if sent_accs and sent_f1s:
                print(f"  Sentiment Classification - Avg Accuracy: {np.mean(sent_accs):.4f}, Avg Macro F1: {np.mean(sent_f1s):.4f}")

def main():
    """Main comparison function"""
    print("Model Performance Comparison Tool")
    print("="*80)
    
    # Load all available results
    results = load_all_results()
    
    if not results:
        print("No model results found!")
        print("Please run the following scripts first:")
        print("  - python bilstm_single_task.py")
        print("  - python bilstm_multi_task.py") 
        print("  - python phobert_single_task.py")
        print("  - python phobert_multitask.py")
        return
    
    # Print comparison summary
    print_comparison_summary(results)
    
    # Create comparison charts
    create_comparison_charts(results)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON COMPLETED!")
    print("="*80)
    print("Comparison charts and summary generated successfully!")

if __name__ == "__main__":
    main()
