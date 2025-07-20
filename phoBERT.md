# PhoBERT Model Evaluation Results

## Tổng quan

Báo cáo này phân tích kết quả đánh giá hiệu suất của hai mô hình PhoBERT được huấn luyện cho bài toán Aspect-based Sentiment Analysis trên dữ liệu đánh giá sản phẩm mỹ phẩm tiếng Việt:

1. **PhoBERT Single-task** (phobert_results_20250721_022006.json)
2. **PhoBERT Multi-task** (phobert_multitask_results_20250721_025942.json)

## Cấu hình mô hình

- **Model**: vinai/phobert-base-v2
- **Max Length**: 128 tokens
- **Timestamp**: 21/07/2025

## Kết quả Aspect Detection

### Bảng so sánh F1-Score

| Aspect | Single-task F1 | Multi-task F1 | Cải thiện |
|--------|----------------|---------------|-----------|
| SMELL | 98.25% | 98.57% | +0.32% |
| TEXTURE | 95.69% | 96.04% | +0.35% |
| COLOUR | 97.35% | 97.82% | +0.47% |
| PRICE | 97.28% | 97.73% | +0.45% |
| SHIPPING | 98.33% | 98.51% | +0.18% |
| PACKING | 95.57% | 95.71% | +0.14% |
| STAYINGPOWER | 96.76% | 96.76% | 0.00% |
| OTHERS | 95.77% | 95.75% | -0.02% |

### Phân tích Aspect Detection

**Điểm mạnh:**
- Hiệu suất tổng thể rất cao (>95% F1-score cho tất cả aspects)
- Mô hình Multi-task cải thiện hiệu suất trong 6/8 aspects
- Cải thiện đáng kể nhất ở COLOUR (+0.47%) và PRICE (+0.45%)

**Điểm cần lưu ý:**
- STAYINGPOWER có hiệu suất tương đương giữa hai mô hình
- OTHERS giảm nhẹ (-0.02%) trong mô hình Multi-task

## Kết quả Sentiment Classification

### Accuracy Comparison

| Aspect | Single-task | Multi-task | Cải thiện |
|--------|-------------|------------|-----------|
| SMELL | 97.78% | 98.03% | +0.25% |
| TEXTURE | 95.01% | 94.95% | -0.06% |
| COLOUR | 93.96% | 95.44% | +1.48% |
| PRICE | 98.71% | 98.71% | 0.00% |
| SHIPPING | 96.43% | 96.49% | +0.06% |
| PACKING | 98.15% | 98.21% | +0.06% |
| STAYINGPOWER | 96.61% | 97.35% | +0.74% |

### Macro F1-Score Comparison

| Aspect | Single-task | Multi-task | Cải thiện |
|--------|-------------|------------|-----------|
| SMELL | 69.35% | 74.07% | +4.72% |
| TEXTURE | 84.30% | 83.01% | -1.29% |
| COLOUR | 78.16% | 82.50% | +4.34% |
| PRICE | 49.11% | 49.09% | -0.02% |
| SHIPPING | 82.95% | 79.97% | -2.98% |
| PACKING | 65.42% | 66.08% | +0.66% |
| STAYINGPOWER | 82.25% | 85.81% | +3.56% |

## Phân tích chi tiết theo từng Aspect

### SMELL
- **Accuracy**: Cải thiện từ 97.78% → 98.03%
- **Macro F1**: Cải thiện đáng kể từ 69.35% → 74.07%
- **Vấn đề**: Lớp Neutral vẫn có hiệu suất rất thấp (F1=0% → 9.52%)

### TEXTURE  
- **Accuracy**: Giảm nhẹ từ 95.01% → 94.95%
- **Macro F1**: Giảm từ 84.30% → 83.01%
- **Điểm mạnh**: Hiệu suất tốt và cân bằng giữa các lớp sentiment

### COLOUR
- **Accuracy**: Cải thiện mạnh từ 93.96% → 95.44%
- **Macro F1**: Cải thiện từ 78.16% → 82.50%
- **Điểm nổi bật**: Cải thiện tốt nhất về accuracy

### PRICE
- **Accuracy**: Không đổi (98.71%)
- **Macro F1**: Gần như không đổi (~49%)
- **Vấn đề**: Lớp Negative và Neutral có hiệu suất rất thấp (F1=0%)

### SHIPPING
- **Accuracy**: Cải thiện nhẹ từ 96.43% → 96.49%
- **Macro F1**: Giảm từ 82.95% → 79.97%
- **Lưu ý**: Lớp Neutral có hiệu suất thấp

### PACKING
- **Accuracy**: Cải thiện từ 98.15% → 98.21%
- **Macro F1**: Cải thiện từ 65.42% → 66.08%
- **Vấn đề**: Lớp Neutral có hiệu suất rất thấp (F1=0%)

### STAYINGPOWER
- **Accuracy**: Cải thiện từ 96.61% → 97.35%
- **Macro F1**: Cải thiện từ 82.25% → 85.81%
- **Điểm mạnh**: Hiệu suất cân bằng giữa các lớp

## Vấn đề chung và Thách thức

### 1. Lớp Neutral
- Hiệu suất kém nhất trong hầu hết các aspects
- Một số aspect có F1=0% cho lớp Neutral (PRICE, PACKING)
- Nguyên nhân: Thiếu dữ liệu huấn luyện, khó phân biệt với các lớp khác

### 2. Lớp Negative  
- Hiệu suất tốt hơn Neutral nhưng vẫn thấp hơn Positive
- Một số aspect có F1=0% cho lớp Negative (PRICE)

### 3. Mất cân bằng dữ liệu
- Lớp "No Aspect" và "Positive" chiếm đa số
- Lớp "Neutral" và "Negative" có ít dữ liệu hơn

## So sánh Single-task vs Multi-task

### Ưu điểm của Multi-task Learning:
1. **Cải thiện aspect detection**: 6/8 aspects có hiệu suất tốt hơn
2. **Tăng accuracy**: Đặc biệt ở COLOUR (+1.48%) và STAYINGPOWER (+0.74%)
3. **Cải thiện macro F1**: SMELL (+4.72%), COLOUR (+4.34%), STAYINGPOWER (+3.56%)
4. **Tận dụng thông tin chung**: Học được đặc trưng chung giữa các task

### Nhược điểm của Multi-task Learning:
1. **Giảm hiệu suất một số aspect**: TEXTURE, SHIPPING
2. **Trade-off**: Cải thiện một số task có thể làm giảm hiệu suất task khác
3. **Phức tạp hơn**: Cần điều chỉnh trọng số giữa các task

## Kết luận và Khuyến nghị

### Kết luận:
1. **Multi-task learning** cho kết quả tốt hơn tổng thể
2. Cả hai mô hình đều đạt hiệu suất cao (>93% accuracy)
3. Vấn đề chính là **lớp Neutral** và **mất cân bằng dữ liệu**

### Khuyến nghị cải thiện:

#### 1. Xử lý mất cân bằng dữ liệu:
- Thu thập thêm dữ liệu cho lớp Neutral và Negative
- Áp dụng kỹ thuật oversampling (SMOTE, ADASYN)
- Sử dụng class weights trong loss function

#### 2. Cải thiện kiến trúc Multi-task:
- Thử nghiệm các kiến trúc sharing khác nhau
- Điều chỉnh trọng số loss giữa aspect detection và sentiment classification
- Áp dụng gradient balancing techniques

#### 3. Kỹ thuật Data Augmentation:
- Paraphrasing cho các lớp thiểu số
- Back-translation
- Synonym replacement

#### 4. Post-processing:
- Threshold tuning cho từng lớp
- Ensemble methods
- Calibration techniques

### Hướng phát triển tiếp theo:
1. Thử nghiệm với các mô hình BERT khác (mBERT, XLM-R)
2. Fine-tuning với domain-specific data
3. Áp dụng few-shot learning cho lớp Neutral
4. Nghiên cứu hierarchical multi-task learning

## Lưu ý về Model Files

Do giới hạn kích thước file của GitHub (100MB), các file model PhoBERT (.pth) không được upload lên repository. Để sử dụng các mô hình:

1. **Huấn luyện lại**: Sử dụng các script được cung cấp để huấn luyện lại mô hình
2. **Download từ nguồn khác**: Liên hệ tác giả để lấy link download model
3. **Sử dụng Git LFS**: Cân nhắc sử dụng Git Large File Storage cho các file lớn

### File structure:
```
models/
├── phobert_aspect_model.pth          # 518MB (excluded)
├── phobert_multitask_model.pth       # 521MB (excluded)  
├── phobert_sentiment_*.pth           # 518MB each (excluded)
├── single_task_*.pth                 # 5.9MB each (included)
└── *_tokenizer.pkl                   # 136KB (included)
```
