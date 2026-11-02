"""Bước 1 — Khai phá dữ liệu (EDA).

Đọc 2 bộ dữ liệu đã chuẩn hoá, thống kê mô tả (số lượng, tỉ lệ lớp, độ dài
văn bản, từ phổ biến), vẽ biểu đồ ra outputs/figures/, và in các nhận xét
dùng cho báo cáo.

Chạy: .venv/bin/python src/01_eda.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

DATASETS = {
    "yt_spam": os.path.join(PROC, "yt_spam.csv"),
    "vi_spam": os.path.join(PROC, "vi_spam.csv"),
}
COLORS = {"yt_spam": "#2e7d32", "vi_spam": "#1565c0"}

notes = []
for name, path in DATASETS.items():
    df = pd.read_csv(path)
    df["len"] = df["text"].astype(str).str.split().str.len()
    n_spam = (df["label"] == "spam").sum()
    n_ham = (df["label"] == "ham").sum()

    print(f"\n===== {name} =====")
    print(f"rows={len(df)}  spam={n_spam} ({n_spam/len(df):.1%})  ham={n_ham} ({n_ham/len(df):.1%})")
    print("do dai (so tu): spam =", round(df.loc[df.label == 'spam', 'len'].mean(), 1),
          "| ham =", round(df.loc[df.label == 'ham', 'len'].mean(), 1))

    # Top từ khoá trong spam (unigram) — đặc trưng nhận dạng
    cv = CountVectorizer(stop_words="english" if name == "yt_spam" else None,
                         max_features=5000)
    X = cv.fit_transform(df.loc[df.label == "spam", "text"])
    freq = X.sum(axis=0).A1
    words = pd.Series(freq, index=cv.get_feature_names_out()).sort_values(ascending=False).head(15)
    print("top tu spam:", list(words.index))

    # ---- Figure 1: phân bố lớp ----
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(["ham", "spam"], [n_ham, n_spam], color=COLORS[name])
    for i, v in enumerate([n_ham, n_spam]):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title(f"Phân bố lớp — {name} ({len(df)} mẫu)")
    ax.set_ylabel("Số mẫu")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}_phan_bo_lop.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 2: phân bố độ dài theo lớp ----
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for lab, col in [("ham", "#9e9e9e"), ("spam", "#c62828")]:
        sub = df.loc[df.label == lab, "len"].clip(upper=100)
        ax.hist(sub, bins=40, alpha=0.55, label=lab, color=col, density=True)
    ax.set_title(f"Phân bố độ dài bình luận (cắt ở 100 từ) — {name}")
    ax.set_xlabel("Số từ")
    ax.set_ylabel("Tỉ trọng")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}_do_dai.png"), dpi=130)
    plt.close(fig)

    notes.append((name, len(df), n_spam, n_ham))

print("\nDa luu bieu do vao", FIG)
