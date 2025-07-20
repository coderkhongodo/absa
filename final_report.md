# Vietnamese Aspect-based Sentiment Analysis: Single-Task vs Multi-Task Learning

## Executive Summary

This report presents a comprehensive comparison between Single-Task Learning and Multi-Task Learning approaches for Vietnamese Aspect-based Sentiment Analysis on beauty product reviews. Both models were implemented using PyTorch with BiLSTM architecture and evaluated on the same dataset.

## Dataset Overview

- **Training Set**: 12,644 samples
- **Validation Set**: 1,585 samples  
- **Test Set**: 1,623 samples
- **Aspects**: 8 categories (smell, texture, colour, price, shipping, packing, stayingpower, others)
- **Sentiment Classes**: 4 categories (no aspect, negative, neutral, positive)

## Model Architectures

### Single-Task Learning
- **Approach**: Separate models for each aspect and sentiment classification
- **Architecture**: BiLSTM + Dense layers for each task
- **Training**: Independent training for each model
- **Total Models**: 15 models (8 aspect detection + 7 sentiment classification)

### Multi-Task Learning
- **Approach**: Unified model with shared representation and task-specific heads
- **Architecture**: Shared BiLSTM + Dense layers → Aspect detection head + Sentiment classification heads
- **Training**: Joint training with combined loss function
- **Total Models**: 1 unified model

## Results Comparison

### Aspect Detection Performance

| Aspect | Single-Task F1 | Multi-Task F1 | Improvement | Single-Task Acc | Multi-Task Acc | Improvement |
|--------|----------------|---------------|-------------|-----------------|----------------|-------------|
| SMELL | 0.9792 | 0.9726 | -0.0066 | 0.9920 | 0.9895 | -0.0025 |
| TEXTURE | 0.9295 | 0.9420 | +0.0125 | 0.9556 | 0.9649 | +0.0092 |
| COLOUR | 0.9720 | 0.9657 | -0.0064 | 0.9747 | 0.9692 | -0.0055 |
| PRICE | 0.9544 | 0.9568 | +0.0024 | 0.9815 | 0.9827 | +0.0012 |
| SHIPPING | 0.9708 | 0.9709 | +0.0001 | 0.9809 | 0.9809 | +0.0000 |
| PACKING | 0.9260 | 0.9551 | +0.0290 | 0.9717 | 0.9834 | +0.0117 |
| STAYINGPOWER | 0.9351 | 0.9466 | +0.0116 | 0.9784 | 0.9815 | +0.0031 |
| OTHERS | 0.9057 | 0.9455 | +0.0398 | 0.9655 | 0.9791 | +0.0136 |

**Overall Averages:**
- **F1-Score**: Single-Task: 0.9466, Multi-Task: 0.9569 (+0.0103)
- **Accuracy**: Single-Task: 0.9750, Multi-Task: 0.9789 (+0.0039)

### Sentiment Classification Performance

| Aspect | Single-Task Acc | Multi-Task Acc | Improvement | Single-Task Macro F1 | Multi-Task Macro F1 | Improvement |
|--------|-----------------|----------------|-------------|---------------------|-------------------|-------------|
| SMELL | 0.8639 | 0.9563 | +0.0923 | 0.5431 | 0.5421 | -0.0010 |
| TEXTURE | 0.8600 | 0.8909 | +0.0309 | 0.7576 | 0.5542 | -0.2034 |
| COLOUR | 0.8707 | 0.9082 | +0.0375 | 0.6065 | 0.4666 | -0.1400 |
| PRICE | 0.9787 | 0.9809 | +0.0022 | 0.3297 | 0.4862 | +0.1564 |
| SHIPPING | 0.9120 | 0.9439 | +0.0319 | 0.7542 | 0.6824 | -0.0718 |
| PACKING | 0.9762 | 0.9760 | -0.0002 | 0.5409 | 0.4811 | -0.0598 |
| STAYINGPOWER | 0.8051 | 0.9051 | +0.1001 | 0.6986 | 0.4551 | -0.2436 |

**Overall Averages:**
- **Accuracy**: Single-Task: 0.8952, Multi-Task: 0.9373 (+0.0421)
- **Macro F1**: Single-Task: 0.6044, Multi-Task: 0.5240 (-0.0804)
- **Weighted F1**: Single-Task: 0.8885, Multi-Task: 0.9241 (+0.0356)

## Key Findings

### Advantages of Multi-Task Learning

1. **Higher Overall Accuracy**: Multi-task learning achieved better accuracy in both aspect detection (+0.39%) and sentiment classification (+4.21%)

2. **Better Generalization**: Significant improvements in challenging aspects:
   - PACKING: +2.90% F1-score improvement
   - OTHERS: +3.98% F1-score improvement
   - STAYINGPOWER sentiment: +10.01% accuracy improvement

3. **Model Efficiency**: Single unified model vs. 15 separate models
   - Reduced memory footprint
   - Faster inference time
   - Easier deployment and maintenance

4. **Shared Representation Learning**: The model learns common features across tasks, leading to better generalization

### Challenges with Multi-Task Learning

1. **Macro F1-Score**: Lower macro F1-score in sentiment classification (-8.04%) indicates difficulty with minority classes

2. **Class Imbalance**: Multi-task learning struggles more with imbalanced classes in sentiment classification

3. **Training Complexity**: Requires careful tuning of loss weights and learning rates

## Technical Implementation

### Training Configuration
- **Epochs**: 30 (with early stopping)
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Functions**: 
  - Aspect Detection: Binary Cross-Entropy
  - Sentiment Classification: Cross-Entropy
- **Early Stopping**: Patience of 7 epochs

### Model Architecture Details
- **Embedding Dimension**: 128
- **Hidden Dimension**: 200 (BiLSTM)
- **Shared Dense Layers**: 128 → 64 → 32
- **Aspect Head**: 32 → 8 (sigmoid)
- **Sentiment Heads**: 33 → 4 (aspect_prob + shared_features)

## Recommendations

### When to Use Multi-Task Learning
1. **Limited Training Data**: When individual tasks have insufficient data
2. **Related Tasks**: When tasks share common underlying patterns
3. **Resource Constraints**: When model size and inference speed are critical
4. **Overall Performance**: When overall accuracy is more important than per-class performance

### When to Use Single-Task Learning
1. **Class Balance**: When dealing with highly imbalanced datasets
2. **Task-Specific Optimization**: When each task requires specialized architecture
3. **Interpretability**: When individual task performance is critical
4. **Debugging**: When easier model debugging and analysis is needed

## Conclusion

Multi-task learning demonstrates superior overall performance for Vietnamese aspect-based sentiment analysis, achieving:
- **+1.03% improvement** in aspect detection F1-score
- **+4.21% improvement** in sentiment classification accuracy
- **Significant efficiency gains** through model unification

However, the approach shows some limitations in handling class imbalance, particularly evident in the macro F1-score decrease for sentiment classification.

For production deployment, **multi-task learning is recommended** due to its superior overall accuracy, efficiency, and practical advantages, with careful attention to class balancing techniques for minority sentiment classes.

## Files Generated

### Model Files
- `models/best_multitask_model.pth` - Trained multi-task model
- `models/multitask_tokenizer.pkl` - Tokenizer for multi-task model
- Individual single-task models in `models/` directory

### Evaluation Results
- `evaluation_results/multitask_results_*.json` - Detailed multi-task results
- `evaluation_results/evaluation_results_*.json` - Detailed single-task results
- `evaluation_results/comparison_summary_*.json` - Comparison summary
- `evaluation_results/*_comparison_*.csv` - Comparison data in CSV format

### Visualizations
- `evaluation_results/aspect_comparison_*.png` - Aspect detection comparison charts
- `evaluation_results/sentiment_comparison_*.png` - Sentiment classification comparison charts
- `evaluation_results/multitask_*_performance_*.png` - Multi-task performance charts

## Code Structure

```
├── single_task_learning.py          # Single-task implementation
├── multi_task_learning_clean.py     # Multi-task implementation  
├── evaluate_single_task.py          # Single-task evaluation
├── evaluate_multitask.py            # Multi-task evaluation
├── compare_models.py                # Model comparison
└── final_report.md                  # This report
```

---

*Report generated on: 2025-07-20*  
*Models trained and evaluated on Vietnamese beauty product review dataset*
