# Vietnamese Aspect-based Sentiment Analysis

This project implements aspect-based sentiment analysis for Vietnamese beauty product reviews using both single-task and multi-task learning approaches with PyTorch.

## Project Structure

```
├── data_preprocessing.py           # Data preprocessing and cleaning
├── single_task_learning.py         # Single-task learning implementation
├── multi_task_learning_clean.py    # Multi-task learning implementation
├── evaluate_multitask.py          # Multi-task model evaluation
├── final_report.md                 # Comprehensive comparison report
├── setup_environment.sh           # Environment setup script
├── requirements.txt                # Python dependencies
├── data_pr/                        # Preprocessed data
├── models/                         # Trained models
└── evaluation_results/             # Evaluation results and charts
```

## Quick Start

### 1. Environment Setup
```bash
bash setup_environment.sh
source venv_absa/bin/activate
```

### 2. Data Preprocessing
```bash
python data_preprocessing.py
```

### 3. Train Models
```bash
# Single-task learning
python single_task_learning.py

# Multi-task learning
python multi_task_learning_clean.py
```

### 4. Evaluate Models
```bash
python evaluate_multitask.py
```

## Model Performance Summary

### Aspect Detection
- **Single-Task**: F1: 0.9466, Accuracy: 0.9750
- **Multi-Task**: F1: 0.9569, Accuracy: 0.9789
- **Improvement**: +1.03% F1, +0.39% Accuracy

### Sentiment Classification
- **Single-Task**: Accuracy: 0.8952, Macro F1: 0.6044
- **Multi-Task**: Accuracy: 0.9373, Macro F1: 0.5240
- **Improvement**: +4.21% Accuracy, -8.04% Macro F1

## Key Features

- Vietnamese text preprocessing and normalization
- Single-task learning: 15 separate models (8 aspect + 7 sentiment)
- Multi-task learning: 1 unified model with shared representation
- Comprehensive evaluation with metrics and visualizations
- Model comparison and analysis

## Aspects Analyzed

1. **smell** - Fragrance/scent
2. **texture** - Product texture/feel
3. **colour** - Color/appearance
4. **price** - Pricing/value
5. **shipping** - Delivery/shipping
6. **packing** - Packaging quality
7. **stayingpower** - Durability/longevity
8. **others** - Other aspects

## Results

Multi-task learning shows superior overall performance with better accuracy and efficiency, while single-task learning excels in handling class imbalance for sentiment classification.

See `final_report.md` for detailed analysis and comparison.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- scikit-learn, pandas, numpy
- matplotlib, seaborn
- underthesea (Vietnamese NLP)

## License

MIT License
