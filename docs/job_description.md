# NLP Assignment - Language Modeling & Tokenization

## Task 1: Data Selection & Exploratory Data Analysis (EDA)

### Đánh giá
Cần thực hiện phân tích đặc điểm ngôn ngữ của từng dataset:

| Dataset | Đặc điểm |
|---|---|
| One Billion Word | Dữ liệu lớn, câu độc lập, không giữ ngữ cảnh dài |
| WikiText-103 | Văn phong Wikipedia, có ngữ cảnh dài, giàu thông tin |
| Text8 | Không có dấu câu, toàn chữ thường |
| Enwik8 | Dữ liệu raw (byte-level), chứa cả ký tự đặc biệt |

Cần phân tích các yếu tố:
- Phân bố từ vựng
- Độ dài câu
- Sự hiện diện của dấu câu
- Tính tự nhiên của ngôn ngữ

## Task 2: Xây dựng thuật toán Prediction

Đề bài tập trung vào **Language Modeling**:

- N-gram models (bigram, trigram)
- RNN / LSTM đơn giản (tự implement, không dùng pre-trained)

### Mục tiêu
Xây dựng mô hình có khả năng:
- Dự đoán từ tiếp theo
- Tính toán Perplexity (PP)

## Task 3: Tokenization (Word, Character, BPE)

### Đánh giá

### Nội dung cần thực hiện
So sánh 3 phương pháp tokenization:

1. **Word-level Tokenization**
   - Ưu điểm:
     - Dễ hiểu, gần với ngôn ngữ tự nhiên
   - Nhược điểm:
     - Vocabulary rất lớn
     - Vấn đề OOV (Out-of-Vocabulary)

2. **Character-level Tokenization**
   - Ưu điểm:
     - Không có OOV
   - Nhược điểm:
     - Sequence rất dài
     - Training chậm hơn

3. **BPE (Byte Pair Encoding)**
   - Ưu điểm:
     - Cân bằng giữa word-level và char-level
     - Giảm OOV
   - Nhược điểm:
     - Cần training tokenizer

## Task 4: Implementation & Evaluation

### Đánh giá

1. **Vocabulary Size**
   - Số lượng token trong từ điển

2. **Sequence Length**
   - Độ dài trung bình của chuỗi sau khi tokenize

3. **Computational Efficiency**
   - Thời gian train
   - Tốc độ xử lý

4. **Perplexity (PP)**
   - Chỉ số quan trọng nhất
   - Đánh giá khả năng dự đoán của mô hình

## Tổng kết

Bài toán tập trung vào:
- So sánh các phương pháp tokenization
- Áp dụng trên nhiều dataset khác nhau
- Đánh giá bằng Perplexity và các chỉ số liên quan

Hướng tiếp cận đúng:
1. EDA kỹ từng dataset
2. Xây dựng language model (N-gram hoặc RNN/LSTM)
3. So sánh Word vs Char vs BPE
4. Đánh giá bằng các metric chuẩn
