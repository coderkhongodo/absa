# Vietnamese Aspect-based Sentiment Analysis for Beauty Products

This project implements aspect-based sentiment analysis for Vietnamese beauty product reviews using multiple approaches including traditional neural networks and state-of-the-art PhoBERT models with both single-task and multi-task learning.

## 🎯 Project Overview

The system analyzes Vietnamese reviews about beauty products (specifically lipstick) to:
- **Detect aspects**: SMELL, TEXTURE, COLOUR, PRICE, SHIPPING, PACKING, STAYINGPOWER, OTHERS
- **Classify sentiment**: Positive, Neutral, Negative, No Aspect

## 📁 Project Structure

```
├── 📄 Core Scripts
│   ├── data_preprocessing.py           # Data preprocessing and cleaning
│   ├── single_task_learning.py         # Traditional single-task learning
│   ├── multi_task_learning_clean.py    # Traditional multi-task learning
│   └── evaluate_multitask.py          # Traditional model evaluation
│
├── 🤖 PhoBERT Models
│   ├── phobert_single_task.py         # PhoBERT single-task implementation
│   ├── phobert_multitask.py           # PhoBERT multi-task implementation
│   ├── evaluate_phobert.py            # PhoBERT single-task evaluation
│   ├── evaluate_phobert_multitask.py  # PhoBERT multi-task evaluation
│   └── compare_models.py              # Model comparison utilities
│
├── 📊 Data & Results
│   ├── data_pr/                       # Preprocessed data
│   ├── models/                        # Trained models (*.pth files)
│   ├── evaluation_results/            # Evaluation results and charts
│   └── phoBERT.md                     # PhoBERT detailed analysis
│
├── 📋 Documentation
│   ├── final_report.md                # Traditional models comparison
│   ├── PROJECT_SUMMARY.md             # Project summary
│   └── requirements.txt               # Python dependencies
│
└── 🛠️ Setup
    └── setup_environment.sh           # Environment setup script
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/coderkhongodo/absa.git
cd absa

# Setup environment
bash setup_environment.sh
source venv_absa/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing
```bash
python data_preprocessing.py
```

### 3. Choose Your Approach

#### Option A: Traditional Neural Networks
```bash
# Single-task learning (15 separate models)
python single_task_learning.py

# Multi-task learning (1 unified model)
python multi_task_learning_clean.py

# Evaluate traditional models
python evaluate_multitask.py
```

#### Option B: PhoBERT Models (Recommended)
```bash
# PhoBERT Single-task learning
python phobert_single_task.py

# PhoBERT Multi-task learning
python phobert_multitask.py

# Evaluate PhoBERT single-task
python evaluate_phobert.py

# Evaluate PhoBERT multi-task
python evaluate_phobert_multitask.py

# Compare all models
python compare_models.py
```

## 📊 Model Performance Summary

### 🏆 PhoBERT Models (Best Performance)

#### Aspect Detection F1-Scores
| Aspect | Single-task | Multi-task | Improvement |
|--------|-------------|------------|-------------|
| SMELL | 98.25% | 98.57% | +0.32% |
| TEXTURE | 95.69% | 96.04% | +0.35% |
| COLOUR | 97.35% | 97.82% | +0.47% |
| PRICE | 97.28% | 97.73% | +0.45% |
| SHIPPING | 98.33% | 98.51% | +0.18% |
| PACKING | 95.57% | 95.71% | +0.14% |
| STAYINGPOWER | 96.76% | 96.76% | 0.00% |
| OTHERS | 95.77% | 95.75% | -0.02% |

#### Sentiment Classification Accuracy
| Aspect | Single-task | Multi-task | Improvement |
|--------|-------------|------------|-------------|
| SMELL | 97.78% | 98.03% | +0.25% |
| TEXTURE | 95.01% | 94.95% | -0.06% |
| COLOUR | 93.96% | 95.44% | +1.48% |
| PRICE | 98.71% | 98.71% | 0.00% |
| SHIPPING | 96.43% | 96.49% | +0.06% |
| PACKING | 98.15% | 98.21% | +0.06% |
| STAYINGPOWER | 96.61% | 97.35% | +0.74% |

### 📈 Traditional Neural Networks
#### Aspect Detection
- **Single-Task**: F1: 0.9466, Accuracy: 0.9750
- **Multi-Task**: F1: 0.9569, Accuracy: 0.9789
- **Improvement**: +1.03% F1, +0.39% Accuracy

#### Sentiment Classification
- **Single-Task**: Accuracy: 0.8952, Macro F1: 0.6044
- **Multi-Task**: Accuracy: 0.9373, Macro F1: 0.5240
- **Improvement**: +4.21% Accuracy, -8.04% Macro F1

## ✨ Key Features

### 🔧 Technical Features
- **Vietnamese text preprocessing** and normalization
- **Multiple model architectures**: Traditional NN + PhoBERT
- **Single-task learning**: Separate models for each task
- **Multi-task learning**: Unified models with shared representation
- **Comprehensive evaluation** with detailed metrics and visualizations
- **Model comparison** and analysis tools

### 🎯 Supported Aspects
1. **SMELL** - Fragrance/scent quality
2. **TEXTURE** - Product texture/feel
3. **COLOUR** - Color/appearance
4. **PRICE** - Pricing/value for money
5. **SHIPPING** - Delivery/shipping experience
6. **PACKING** - Packaging quality
7. **STAYINGPOWER** - Durability/longevity
8. **OTHERS** - Other miscellaneous aspects

### 📱 Sentiment Classes
- **Positive**: Favorable opinion
- **Neutral**: Neutral/mixed opinion
- **Negative**: Unfavorable opinion
- **No Aspect**: Aspect not mentioned

## 🔍 Detailed Analysis

### PhoBERT Models
- **Best overall performance** with >95% F1-score for aspect detection
- **Multi-task learning** shows improvements in 6/8 aspects
- **Significant improvements** in COLOUR (+1.48% accuracy) and SMELL (+4.72% macro F1)
- **Challenge**: Neutral class classification remains difficult

### Traditional Models
- **Good baseline performance** but lower than PhoBERT
- **Multi-task learning** shows better aspect detection
- **Single-task learning** better handles class imbalance

📖 **Detailed Analysis**: See `phoBERT.md` for comprehensive PhoBERT analysis and `final_report.md` for traditional models comparison.

## 💻 System Requirements

### Hardware
- **RAM**: Minimum 8GB, Recommended 16GB+
- **GPU**: Optional but recommended for PhoBERT training (CUDA compatible)
- **Storage**: At least 5GB free space

### Software
- **Python**: 3.8+ (tested on 3.8, 3.9, 3.10)
- **Operating System**: Linux, macOS, Windows

### Dependencies
```bash
# Core ML libraries
torch>=1.9.0
transformers>=4.0.0
scikit-learn>=1.0.0

# Data processing
pandas>=1.3.0
numpy>=1.21.0
underthesea>=1.3.0  # Vietnamese NLP

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Utilities
tqdm>=4.62.0
pickle5>=0.0.11
```

## 🗂️ Data Format

### Input Data Structure
```csv
text,SMELL,TEXTURE,COLOUR,PRICE,SHIPPING,PACKING,STAYINGPOWER,OTHERS,SMELL_sentiment,TEXTURE_sentiment,...
"Son màu đẹp lắm",0,0,1,0,0,0,0,0,"","","Positive","","","","",""
```

### Aspect Labels
- `0`: Aspect not mentioned
- `1`: Aspect mentioned

### Sentiment Labels
- `"No Aspect"`: Aspect not mentioned
- `"Positive"`: Positive sentiment
- `"Neutral"`: Neutral sentiment
- `"Negative"`: Negative sentiment

## 🚨 Important Notes

### Model Files
⚠️ **Large model files** (PhoBERT *.pth files, ~518MB each) are **not included** in the repository due to GitHub's 100MB file size limit.

**To use pre-trained models:**
1. **Train from scratch** using provided scripts
2. **Contact authors** for model download links
3. **Use Git LFS** for large file storage (optional)

### File Structure After Training
```
models/
├── 📁 Traditional Models (included)
│   ├── single_task_*.pth           # ~6MB each
│   └── *_tokenizer.pkl             # ~136KB
│
└── 📁 PhoBERT Models (excluded from repo)
    ├── phobert_aspect_model.pth    # ~518MB
    ├── phobert_multitask_model.pth # ~521MB
    └── phobert_sentiment_*.pth     # ~518MB each
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{tran-etal-2022-aspect,
    title = "Aspect-based Sentiment Analysis for {V}ietnamese Reviews about Beauty Product on {E}-commerce Websites",
    author = "Tran, Quang-Linh and Le, Phan Thanh Dat and Do, Trong-Hop",
    booktitle = "Proceedings of the 36th Pacific Asia Conference on Language, Information and Computation",
    month = oct,
    year = "2022",
    address = "Manila, Philippines",
    publisher = "De La Salle University",
    url = "https://aclanthology.org/2022.paclic-1.84",
    pages = "767--776",
}
```

## 📞 Contact

- **Repository**: [https://github.com/coderkhongodo/absa](https://github.com/coderkhongodo/absa)
- **Issues**: [GitHub Issues](https://github.com/coderkhongodo/absa/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repository if you find it helpful!**
