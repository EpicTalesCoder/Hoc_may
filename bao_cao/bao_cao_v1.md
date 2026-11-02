# BÁO CÁO MÔN HỌC MÁY

## Đề tài: Phân loại spam/ham các bình luận của sinh viên PTIT

**Sinh viên thực hiện:** mq (EpicTalesCoder)
**Lớp / GVHD:** …………………………………………
**Repo mã nguồn:** https://github.com/EpicTalesCoder/Hoc_may
**Phiên bản báo cáo:** v1

---

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Dữ liệu](#2-dữ-liệu)
   - 2.1. Nguồn dữ liệu
   - 2.2. Tiền xử lý
   - 2.3. Khai phá dữ liệu (EDA)
3. [Phương pháp](#3-phương-pháp)
   - 3.1. Trích xuất đặc trưng
   - 3.2. Các mô hình phân loại
   - 3.3. Quy trình huấn luyện và đánh giá
4. [Kết quả](#4-kết-quả)
   - 4.1. So sánh các cấu hình (cross-validation)
   - 4.2. Đánh giá trên tập test
   - 4.3. Phân tích lỗi
5. [Hạn chế của hệ thống hiện tại](#5-hạn-chế-của-hệ-thống-hiện-tại)
6. [Các phần cần cải thiện (dành cho phiên bản sau)](#6-các-phần-cần-cải-thiện-dành-cho-phiên-bản-sau)
7. [Kết luận](#7-kết-luận)
8. [Tài liệu tham khảo](#8-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

Bình luận rác (spam) — quảng cáo, lừa đảo, câu like/subscribe, tin nhắn
trúng thưởng — làm nhiễu không gian trao đổi trực tuyến của sinh viên trên
các nền tảng mạng xã hội, nhóm học tập và diễn đàn trường lớp. Bài toán đặt
ra: từ nội dung văn bản của một bình luận, dự đoán nhãn **spam** hay
**ham** (bình luận hợp lệ).

Đây là bài toán **phân loại nhị phân trên dữ liệu văn bản** — dữ liệu phi
cấu trúc, không có đặc trưng số sẵn có, nên cần pipeline: tiền xử lý →
trích xuất đặc trưng (Bag-of-Words, TF-IDF) → huấn luyện mô hình học máy →
đánh giá. Bài làm thực hiện trọn vẹn pipeline này trên 2 bộ dữ liệu thực,
so sánh 10 cấu hình (2 cách biểu diễn × 5 mô hình), tuning tham số và phân
tích lỗi chi tiết.

**Đáp ứng yêu cầu môn học** (theo `Yeucau.md`):

| Yêu cầu | Chỗ đáp ứng |
|---|---|
| Dữ liệu tự tìm kiếm, tối thiểu 2 bộ | 2 bộ dữ liệu công khai (mục 2.1) |
| Phương pháp tổng hợp kiến thức đã học | 5 mô hình + 2 cách biểu diễn đặc trưng (mục 3) |
| Phân tích kết quả, lỗi của mô hình | Mục 4.2–4.3, file lỗi trong `outputs/metrics/` |
| Điểm cộng: kiến thức mới nhất về ML | Đề xuất cụ thể ở mục 6.4 (PhoBERT, char n-grams) |
| Bắt buộc: in quyển báo cáo bìa mềm | Bản in từ file này, chèn ảnh `outputs/figures/` |

## 2. Dữ liệu

### 2.1. Nguồn dữ liệu

Bộ dữ liệu bình luận của sinh viên PTIT không được công khai, nên bài làm
sử dụng 2 bộ dữ liệu thực, có gán nhãn, đại diện cho 2 môi trường bình luận:

| # | Bộ dữ liệu | Nguồn | Bản gốc | Sau làm sạch | Spam | Ham | License |
|---|---|---|---|---|---|---|---|
| 1 | `yt_spam` — YouTube Spam Collection | [UCI ML Repository, dataset 380](https://archive.ics.uci.edu/dataset/380/youtube+spam+collection) | 1.956 bình luận YouTube (5 video nhạc) | **1.760** | 841 (47,8%) | 919 (52,2%) | CC BY 4.0 |
| 2 | `vi_spam` — Vietnamese SMS Spam Detection | [Kaggle](https://www.kaggle.com/datasets/trnminhmn/vietnamese-sms-spam-detection-dataset) | 5.819 tin nhắn tiếng Việt | **4.555** | 732 (16,1%) | 3.823 (83,9%) | — |

- Bộ 1: bình luận YouTube tiếng Anh, gần cân bằng lớp (47,8% spam).
- Bộ 2: tin nhắn/bình luận tiếng Việt, lệch lớp nặng (1 spam : 5 ham) —
  đúng thực tế các group sinh viên, nơi spam là thiểu số.
- Tổng cộng **6.315 mẫu** sau làm sạch, schema thống nhất `text, label`.

### 2.2. Tiền xử lý

Thực hiện trong `src/00_prepare_data.py` (chạy lại được):

1. **Gộp nguồn**: 5 file CSV YouTube gộp thành 1, thêm cột `source_video`.
2. **Chuẩn hoá nhãn**: bộ tiếng Việt có 250 dòng nhãn `1` (lỗi gán nhãn của
   dataset) — theo mô tả của tác giả, nhóm này mang đặc trưng spam nên được
   quy về `spam`, không bỏ đi.
3. **Làm sạch**: bỏ dòng null (449 dòng tiếng Việt bị text rỗng), bỏ trùng
   lặp **theo nội dung** — yt: −196 dòng; vi: −1.264 dòng (spam thường bị
   đăng lặp lại; việc bỏ trùng trước khi chia train/test tránh hiện tượng
   một bản sao nằm ở cả hai tập làm điểm test đẹp giả — data leakage).
4. Kết quả xuất ra `data/processed/{yt_spam.csv, vi_spam.csv}`.

### 2.3. Khai phá dữ liệu (EDA)

Thực hiện trong `src/01_eda.py`; biểu đồ trong `outputs/figures/`:

- **Phân bố lớp** (`yt_spam_phan_bo_lop.png`, `vi_spam_phan_bo_lop.png`):
  yt_spam cân bằng 47,8/52,2; vi_spam lệch 16,1/83,9.
- **Độ dài bình luận** (`*_do_dai.png`): spam YouTube dài hơn hẳn ham
  (**22,6 so với 9,3 từ/bình luận**) — spam chèn link, kêu gọi subscribe;
  ham thường rất ngắn ("nice song!"). Ở bộ tiếng Việt chênh lệch nhỏ hơn
  (17,0 so với 11,8 từ).
- **Từ khoá đặc trưng của lớp spam** (CountVectorizer trên lớp spam):
  - YouTube: *check, subscribe, channel, http, www, com, like, guys…*
    → mô-típ câu like/subscribe/chèn link.
  - Tiếng Việt: *nhận, tiền, vay, giải, gọi, phí, ngay…*
    → mô-típ trúng thưởng, cho vay, mời gọi hành động.

Kết luận EDA: 2 bộ khác nhau cả về ngôn ngữ, độ lệch lớp lẫn "phong cách"
spam — là thách thức tốt để kiểm tra mức độ tổng quát của mô hình.

## 3. Phương pháp

### 3.1. Trích xuất đặc trưng

So sánh 2 cách biểu diễn văn bản kinh điển:

| Biểu diễn | Thiết lập |
|---|---|
| **BoW** (Bag-of-Words) | Đếm từ, n-gram (1,2), min_df=2, max_df=0.95 |
| **TF-IDF** | n-gram (1,2), min_df=2, max_df=0.95, sublinear_tf |

### 3.2. Các mô hình phân loại

5 mô hình thuộc các họ khác nhau (tổng hợp kiến thức đã học):

| Mô hình | Tham số gốc | Ý nghĩa |
|---|---|---|
| Naive Bayes (MultinomialNB) | alpha=1.0 | Chuẩn mực cho văn bản, giả thiết độc lập |
| Logistic Regression | max_iter=1000 | Baseline tuyến tính, có xác suất |
| SVM tuyến tính (LinearSVC) | — | Biên tối đại, mạnh trên văn bản thưa |
| K-Nearest Neighbors | k=5 | Học phi tham số, so sánh "gần nhất" |
| Random Forest | 200 cây | Ensemble bagging, phi tuyến |

### 3.3. Quy trình huấn luyện và đánh giá

Quy trình chống rò rỉ dữ liệu (data leakage), trong `src/02_train_eval.py`:

1. Tách **train/test stratified 80/20, random_state=42** — tỷ lệ lớp được
   giữ nguyên ở cả hai tập; **tập test giữ nguyên và chỉ chạm đúng 1 lần ở
   bước cuối**.
2. Trên tập train: **5-fold cross-validation** cho 10 tổ hợp
   (2 biểu diễn × 5 mô hình), thang đo chọn **F1 của lớp spam** (thích hợp
   khi lớp dương là thiểu số).
3. Cấu hình tốt nhất được **tuning tham số bằng GridSearchCV** (5-fold,
   vẫn chỉ trên train): alpha của NB, C của LR/SVM, k của KNN, số cây của RF.
4. Fit lại trên toàn bộ train → đánh giá trên test: Accuracy, Precision,
   Recall, F1 + confusion matrix.

## 4. Kết quả

### 4.1. So sánh các cấu hình (CV F1 trên tập train)

| # | yt_spam | CV F1 | | vi_spam | CV F1 |
|---|---|---|---|---|---|
| 1 | BoW + RandomForest | **0,9395 ± 0,018** | | BoW + NaiveBayes | **0,8611 ± 0,013** |
| 2 | TFIDF + RandomForest | 0,9387 ± 0,022 | | TFIDF + LinearSVM | 0,8611 ± 0,013 |
| 3 | TFIDF + LinearSVM | 0,9378 ± 0,022 | | BoW + LinearSVM | 0,8351 ± 0,023 |
| 4 | BoW + LinearSVM | 0,9355 ± 0,019 | | BoW + LogisticRegression | 0,8263 ± 0,023 |
| 5 | BoW + LogisticRegression | 0,9347 ± 0,011 | | BoW + RandomForest | 0,8173 ± 0,022 |
| 6 | TFIDF + LogisticRegression | 0,9322 ± 0,025 | | TFIDF + RandomForest | 0,8140 ± 0,032 |
| 7 | BoW + NaiveBayes | 0,9276 ± 0,024 | | TFIDF + NaiveBayes | 0,7677 ± 0,042 |
| 8 | TFIDF + NaiveBayes | 0,9188 ± 0,023 | | TFIDF + LogisticRegression | 0,7632 ± 0,042 |
| 9 | BoW + KNN | 0,8234 ± 0,027 | | BoW + KNN | 0,6242 ± 0,062 |
| 10 | TFIDF + KNN | 0,4374 ± 0,058 | | TFIDF + KNN | 0,3494 ± 0,065 |

(Số liệu đầy đủ: `outputs/metrics/summary.csv`.)

Nhận xét:
- **KNN yếu nhất ở cả 2 bộ** — đặc biệt tệ với TF-IDF (F1 0,44/0,35):
  không gian TF-IDF thưa, chuẩn hoá chiều làm khoảng cách mất ý nghĩa.
- **Bộ lệch lớp (vi_spam) ưu tiên mô hình xác suất/tuyến tính**
  (NaiveBayes, LinearSVM); **bộ cân bằng (yt_spam) hưởng lợi từ ensemble**
  (RandomForest) — lựa chọn mô hình phải đi theo đặc điểm dữ liệu.
- BoW và TF-IDF chênh nhau không đáng kể với đa số mô hình.

### 4.2. Đánh giá trên tập test (mô hình tốt nhất sau tuning)

| Bộ dữ liệu | Cấu hình | Tham số tốt nhất | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) |
|---|---|---|---|---|---|---|
| yt_spam | BoW + Random Forest | n_estimators=200 | **97,73%** | 0,982 | 0,970 | **0,976** |
| vi_spam | BoW + Naive Bayes | alpha=0.5 | **95,94%** | 0,943 | 0,795 | **0,863** |

(Cấu hình: `outputs/metrics/*_metrics.json`; confusion matrix:
`outputs/figures/*_confusion_matrix.png`.)

- Cả 2 bộ đều đạt độ chính xác > 95% trên dữ liệu chưa từng thấy.
- yt_spam: cân bằng giữa precision và recall (0,982/0,970) — ít báo nhầm
  lẫn ít sót.
- vi_spam: precision cao (0,943) nhưng recall thấp (0,795) — mô hình bỏ
  sót ~20% spam. Nguyên nhân chính: lớp spam chỉ chiếm 16,1% (lệch 1:5)
  và mô hình chưa có cơ chế bù lệch lớp.

### 4.3. Phân tích lỗi

Liệt kê đầy đủ mẫu test dự đoán sai trong
`outputs/metrics/{yt,vi}_spam_errors.csv` (8/352 và 37/911 mẫu).

**spam → ham (sót spam):**
- Spam rất ngắn, không từ khoá quảng cáo: *"LÀM TỐT!"* (tiếng Việt),
  *"subscribed :) btw you have a good style…"* (YouTube).
- Spam dùng cách diễn đạt mới không xuất hiện trong tập train — hạn chế
  cố hữu của biểu diễn đếm từ (BoW): từ ngoài từ điển train mất hết ý nghĩa.

**ham → spam (báo nhầm):**
- Ham chứa link/URL hoặc trùng mô-típ từ khoá spam: *"Share this video..
  This song can beat PSY - Gangnam Style!"* — ý kiến thật nhưng chứa động
  từ chia sẻ; *"Xin chúc mừng quặng mo owo re wa."* — câu vô nghĩa gây
  nghi trúng thưởng.
- Nếu lỗi spam→ham nằm ở văn bản biên thì lỗi ham→spam cho thấy đặc trưng
  thuần nội dung từ chưa đủ — cần đặc trưng hành vi (tần suất đăng, tỷ lệ
  URL, tài khoản) như các hệ thống anti-spam thực tế.

## 5. Hạn chế của hệ thống hiện tại

1. **Chưa xử lý lệch lớp**: vi_spam lệch 1:5, recall spam chỉ 0,795;
   pipeline chưa dùng class_weight hay điều chỉnh ngưỡng.
2. **Đặc trưng "vắt chữ"**: BoW/TF-IDF băm từ, không hiểu ngữ cảnh;
   spam viết lái kiểu "fr33", "trúng thư ởng" thoát qua dễ dàng.
3. **Dữ liệu tiếng Việt là SMS dịch máy** từ tiếng Anh, một số câu dịch
   gượng; chưa phản ánh 100% văn phong bình luận của sinh viên.
4. **Đánh giá trên 2 bộ độc lập**, chưa có thí nghiệm tổng quát hoá
   chéo (train trên bộ này, test trên bộ kia).
5. **Mô hình lưu (.joblib) chưa có công cụ dùng lại** (demo dự đoán câu
   mới) cho người chấm chạy thử.

## 6. Các phần cần cải thiện (dành cho phiên bản sau)

Danh sách ưu tiên theo mức độ "ăn điểm / công sức" — sẽ triển khai ở
`bao_cao_v2`:

- [ ] **6.1. Xử lý lệch lớp** (ưu tiên cao nhất):
  `class_weight="balanced"` cho NB/LR/SVM; điều chỉnh ngưỡng dự đoán
  (threshold moving) chọn theo F1 trên CV; mục tiêu: tăng recall spam
  vi_spam từ 0,795 lên > 0,90 mà precision không giảm quá mạnh.
- [ ] **6.2. Giải thích mô hình**: in top đặc trưng theo lớp (hệ số
  LR/NB, feature importance RF) — ví dụ "nhận tiền +2.1, vay +1.8,
  subscribe +2.5" — biến hộp đen thành bảng bằng chứng.
- [ ] **6.3. Đồ thị đánh giá đầy đủ**: ROC + Precision-Recall curve
  (PR quan trọng hơn ROC khi lệch lớp); bảng test metric cho TẤT CẢ
  5 mô hình (hiện chỉ có model thắng).
- [ ] **6.4. Điểm cộng — kiến thức ML mới** (theo yêu cầu môn):
  - char n-grams TF-IDF (chống spam viết lái "fr33", "v@y");
  - word segmentation tiếng Việt (underthesea/VnCoreNLP) trước vector hoá;
  - tinh chỉnh PhoBERT/ViSoBERT trên tập nhỏ, so sánh "classical vs
    modern" trong 1 bảng.
- [ ] **6.5. Thí nghiệm cross-dataset**: train yt → test vi và ngược lại;
  phân tích đặc trưng nào tổng quát (URL, số điện thoại, "free/nhận")
  và cái nào phụ thuộc ngôn ngữ.
- [ ] **6.6. Công bố hạn chế dữ liệu một cách chủ động**: thí nghiệm nhỏ
  chứng minh ảnh hưởng của văn bản dịch máy (lọc nhóm câu dịch gượng, so
  sánh điểm số).
- [ ] **6.7. Vận hành**: script demo `predict.py` (nhập câu mới → nhãn),
  notebook end-to-end, `requirements.txt` pin version chính xác.

## 7. Kết luận

Bài làm xây dựng pipeline hoàn chỉnh cho bài toán phân loại spam/ham
bình luận: **2 bộ dữ liệu thực** (1.760 + 4.555 mẫu sau làm sạch, tổng
6.315), **so sánh 10 cấu hình** (2 cách biểu diễn × 5 mô hình) bằng
5-fold cross-validation chống rò rỉ dữ liệu, tuning GridSearchCV và đánh
giá khách quan trên tập test tách riêng. Kết quả tốt nhất đạt **F1 0,976**
(YouTube, BoW + Random Forest) và **0,863** (tiếng Việt, BoW + Naive
Bayes). Phân tích lỗi chỉ rõ ranh giới của mô hình: yếu ở văn bản biên
(spam ngắn không từ khoá, ham chứa URL) và ở lớp thiểu số lệch 1:5 —
làm cơ sở cho các cải thiện ở mục 6.

## 8. Tài liệu tham khảo

1. Alberto, T. & Lochter, J. (2015). *YouTube Spam Collection*.
   UCI Machine Learning Repository. DOI: 10.24432/C58885. (CC BY 4.0)
2. *Vietnamese SMS Spam Detection Dataset*. Kaggle.
   https://www.kaggle.com/datasets/trnminhmn/vietnamese-sms-spam-detection-dataset
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*.
   JMLR 12, 2825–2830.
4. McKinney, W. (2010). *Data Structures for Statistical Computing in
   Python*. Proc. SciPy.
5. Pham, N. A. & Nguyen, A. V. (2020). *PhoBERT: Pre-trained language
   models for Vietnamese*. EMNLP Findings. (dự kiến cho cải thiện 6.4)
