# Báo cáo môn Học máy

## Đề tài: Phân loại spam/ham các bình luận của sinh viên PTIT

> Bản thảo phục vụ báo cáo cuối kỳ (yêu cầu in bìa mềm — xem `Yeucau.md`).
> Toàn bộ số liệu trong báo cáo này sinh tự động từ pipeline trong `src/`,
> chạy lại được 100%: `00_prepare_data.py` → `01_eda.py` → `02_train_eval.py`.

---

## 1. Giới thiệu

Bình luận rác (spam) — quảng cáo, lừa đảo, câu like/subscribe, tin nhắn
trúng thưởng — làm nhiễu không gian trao đổi của sinh viên trên mạng xã hội.
Bài toán đặt ra: từ nội dung văn bản của một bình luận, dự đoán nhãn
**spam** hay **ham** (bình luận hợp lệ). Đây là bài toán phân loại nhị phân
trên dữ liệu văn bản — dữ liệu không cấu trúc, cần trích xuất đặc trưng
trước khi đưa vào mô hình học máy.

Vì bộ dữ liệu bình luận của sinh viên PTIT không được công khai, bài làm
dùng **2 bộ dữ liệu thực, có gán nhãn** (đạt yêu cầu "tối thiểu 2 bộ dữ
liệu") đại diện cho 2 môi trường bình luận:

| Bộ dữ liệu | Nguồn | Bản gốc | Sau làm sạch | Spam | Ham |
|---|---|---|---|---|---|
| `yt_spam` | YouTube Spam Collection, UCI (CC BY 4.0) | 1.956 | **1.760** | 841 (47,8%) | 919 (52,2%) |
| `vi_spam` | Vietnamese SMS Spam Detection, Kaggle | 5.819 | **4.555** | 732 (16,1%) | 3.823 (83,9%) |

Bộ 1 là bình luận YouTube tiếng Anh (5 video nhạc phổ biến), cân bằng lớp.
Bộ 2 là tin nhắn/bình luận tiếng Việt, lệch lớp nặng (1 spam : 5 ham) —
đúng thực tế các group sinh viên, nơi spam là thiểu số.

## 2. Dữ liệu và tiền xử lý (`src/00_prepare_data.py`)

- **Gộp nguồn**: 5 file CSV YouTube gộp thành 1, thêm cột `source_video`.
- **Chuẩn hoá nhãn**: bộ tiếng Việt có nhãn lỗi (`1`) được quy về `spam`
  theo mô tả của tác giả dataset.
- **Làm sạch**: bỏ dòng null, bỏ trùng lặp theo nội dung
  (yt: −196 dòng; vi: −1.264 dòng — spam thường bị đăng lặp lại, giữ 1 bản).
- Kết quả chuẩn hoá về chung schema `text, label` cho cả 2 bộ.

**Thống kê khám phá (`src/01_eda.py`, biểu đồ trong `outputs/figures/`):**
- Độ dài: spam YouTube dài hơn hẳn ham (22,6 so với 9,3 từ/bình luận) —
  spam chèn link, kêu gọi subscribe; ham thường ngắn ("nice song!").
- Từ đặc trưng spam (YouTube): *check, subscribe, channel, http, www…*
- Từ đặc trưng spam (tiếng Việt): *nhận, tiền, vay, giải, gọi, phí* —
  mô-típ trúng thưởng, cho vay, mời gọi hành động.

## 3. Phương pháp (tổng hợp kiến thức đã học — `src/02_train_eval.py`)

**Trích xuất đặc trưng** (so sánh 2 cách biểu diễn):
- **BoW** — Bag-of-Words đếm từ, n-gram (1,2), min_df=2, max_df=0.95.
- **TF-IDF** — trọng số tần suất nghịch đảo, sublinear_tf.

**5 mô hình phân loại**: Naive Bayes (MultinomialNB), Logistic Regression,
SVM tuyến tính (LinearSVC), K-Nearest Neighbors (k=5), Random Forest
(200 cây).

**Quy trình đánh giá chống rò rỉ dữ liệu (data leakage):**
1. Tách **train/test stratified 80/20, random_state=42** — tập test giữ
   nguyên, chỉ chạm đúng 1 lần ở bước cuối.
2. Trên tập train: **5-fold cross-validation** cho mọi tổ hợp
   (2 biểu diễn × 5 mô hình = 10 cấu hình), chọn theo **F1 lớp spam**.
3. Cấu hình tốt nhất được **tuning tham số** bằng GridSearchCV (vẫn chỉ
   trên train): alpha của NB, C của LR/SVM, k của KNN, số cây của RF.
4. Fit lại trên toàn bộ train → đánh giá trên test: Accuracy, Precision,
   Recall, F1 + confusion matrix.

## 4. Kết quả

### 4.1. So sánh mô hình (CV F1 trên tập train)

| # | yt_spam (BoW/TFIDF + mô hình) | CV F1 | | vi_spam | CV F1 |
|---|---|---|---|---|---|
| 1 | BoW + RandomForest | **0,9395** | | BoW + NaiveBayes | **0,8611** |
| 2 | TFIDF + RandomForest | 0,9387 | | TFIDF + LinearSVM | 0,8611 |
| 3 | TFIDF + LinearSVM | 0,9378 | | BoW + LinearSVM | 0,8351 |
| 4 | BoW + LinearSVM | 0,9355 | | BoW + LogisticRegression | 0,8263 |
| 5 | BoW + LogisticRegression | 0,9347 | | BoW + RandomForest | 0,8173 |

(KNN yếu nhất ở cả 2 bộ — F1 0,82 / 0,62 và chỉ 0,44 với TFIDF, vì không
gian TF-IDF thưa làm khoảng cách cosine mất ý nghĩa.)

### 4.2. Đánh giá trên tập test (mô hình tốt nhất sau tuning)

| Bộ dữ liệu | Cấu hình tốt nhất | Tham số | Accuracy | Precision | Recall | F1 (spam) |
|---|---|---|---|---|---|---|
| yt_spam | BoW + Random Forest | n_estimators=200 | **97,73%** | 0,982 | 0,970 | **0,976** |
| vi_spam | BoW + Naive Bayes | alpha=0.5 | **95,94%** | 0,943 | 0,795 | **0,863** |

Nhận xét:
- Cả hai bộ đều đạt độ chính xác > 95% trên dữ liệu chưa từng thấy.
- Bộ tiếng Việt có recall thấp hơn (0,795): mô hình bỏ sót ~20% spam —
  hệ quả của dữ liệu lệch lớp 1:5 và spam tiếng Việt ngắn, khó đoán
  (ví dụ "LÀM TỐT!" bị nhầm là ham).
- BoW + Naive Bayes thắng ở bộ lệch lớp tiếng Việt; Random Forest thắng
  ở bộ cân bằng tiếng Anh — lựa chọn mô hình phải đi theo dữ liệu.

### 4.3. Phân tích lỗi (file `{yt,vi}_spam_errors.csv`)

- **spam → ham** (sót): spam rất ngắn, không từ khoá quảng cáo
  ("LÀM TỐT!", "subscribed :) btw you have a good style…"); spam dùng
  cách diễn đạt mới không có trong train.
- **ham → spam** (nhầm): ham chứa link/URL hoặc từ khoá trùng mô-típ spam
  ("Share this video.. This song can beat PSY - Gangnam Style!" là ý kiến
  thật nhưng chứa mô-típ chia sẻ).
- Kết luận: lỗi tập trung vào văn bản biên — bình luận ngắn và văn bản
  chứa URL — đúng hạn chế của mô hình dùng duy nhất đặc trưng nội dung từ.

## 5. Hạn chế và hướng phát triển

- **Điểm cộng — kiến thức ngoài chương trình**: hiện trạng là mô hình
  classical ML + đặc trưng thủ công. Hướng phát triển: (1) word embedding
  tiếng Việt (PhoW2Vec/PhoBERT) để nắm ngữ cảnh thay vì đếm từ; (2) xử lý
  mất cân bằng lớp (SMOTE/class_weight) để tăng recall spam tiếng Việt;
  (3) thêm đặc trưng hành vi (tần suất đăng, tỷ lệ URL) như các hệ thống
  anti-spam thực tế.
- Dữ liệu tiếng Việt hiện là tin nhắn SMS, chưa phải bình luận group
  sinh viên PTIT — khi có dữ liệu chuẩn của đề tài, pipeline chạy lại
  nguyên vẹn (chỉ thay file trong `data/processed/`).
- Báo cáo in bìa mềm: chèn ảnh từ `outputs/figures/`, bảng từ
  `outputs/metrics/summary.csv`.

## 6. Kết luận

Bài làm xây dựng pipeline hoàn chỉnh cho bài toán phân loại spam/ham
bình luận: 2 bộ dữ liệu thực (1.760 + 4.555 mẫu sau làm sạch), so sánh
10 cấu hình (2 cách biểu diễn × 5 mô hình) bằng cross-validation, tuning
và đánh giá khách quan trên tập test tách riêng. Kết quả tốt nhất đạt
F1 0,976 (YouTube) và 0,863 (tiếng Việt), kèm phân tích lỗi chỉ rõ ranh
giới của mô hình và hướng cải thiện.
