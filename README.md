# Hoc_may — Phân loại spam/ham bình luận của SV PTIT

Đề tài môn Học máy: phân loại spam/ham các bình luận của sinh viên PTIT
(xem `Detai.md` cho đề tài, `Yeucau.md` cho yêu cầu môn học).

## Cấu trúc

- `src/` — code pipeline (chuẩn hoá dữ liệu, tiền xử lý, huấn luyện, đánh giá)
- `data/` — dữ liệu; raw bỏ qua git, chạy lại `src/00_prepare_data.py` để tạo `data/processed/`
- `outputs/figures` — biểu đồ, `outputs/metrics` — kết quả đánh giá (.json/.csv), `outputs/models` — model đã lưu
- `bao_cao/` — bản thảo báo cáo cuối kỳ (in bìa mềm theo yêu cầu môn)

## Dữ liệu (2 bộ, đạt yêu cầu tối thiểu của môn)

1. **YouTube Spam Collection (UCI, CC BY 4.0)** — 1.956 bình luận thật
   spam/ham từ 5 video, tải từ UCI: https://archive.ics.uci.edu/dataset/380/youtube+spam+collection
2. **Vietnamese SMS Spam Detection (Kaggle)** — 5.819 tin nhắn tiếng Việt
   spam/ham (Kagglehub id: `trnminhmn/vietnamese-sms-spam-detection-dataset`).

Vì bộ chuẩn của SV PTIT không công khai, 2 bộ trên mô phỏng 2 môi trường
bình luận: YouTube (tiếng Anh) và tin nhắn/mạng xã hội (tiếng Việt).

## Chạy pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/00_prepare_data.py   # gộp + làm sạch -> data/processed/
.venv/bin/python src/01_eda.py            # EDA + biểu đồ -> outputs/figures/
.venv/bin/python src/02_train_eval.py     # huấn luyện + đánh giá -> outputs/
```

## Kết quả

Xem `outputs/metrics/summary.csv` (bảng so sánh 5 mô hình × 2 bộ dữ liệu)
và `bao_cao/bao_cao.md`.
