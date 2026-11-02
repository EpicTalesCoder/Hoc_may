"""Bước 0 — Chuẩn hoá dữ liệu.

Gộp 5 file YouTube Spam Collection (UCI) thành một file, chuẩn hoá nhãn
bộ dữ liệu tiếng Việt (ham/spam/1 -> ham/spam), xoá trùng lặp và null,
xuất ra data/processed/{yt_spam.csv, vi_spam.csv}.

Chạy: .venv/bin/python src/00_prepare_data.py
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_YT = os.path.join(ROOT, "data", "raw", "yt_spam")
RAW_VI = os.path.join(ROOT, "data", "raw", "vi_sms")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

# ---------- 1) YouTube Spam Collection ----------
files = sorted(glob.glob(os.path.join(RAW_YT, "Youtube0*.csv")))
dfs = []
for f in files:
    d = pd.read_csv(f)
    d["source_video"] = os.path.basename(f).split("-")[-1].replace(".csv", "")
    dfs.append(d)
yt = pd.concat(dfs, ignore_index=True)

n0_yt = len(yt)
yt = yt.dropna(subset=["CONTENT"])
yt = yt.drop_duplicates(subset=["CONTENT"]).reset_index(drop=True)
yt = yt.rename(columns={"CONTENT": "text", "CLASS": "label"})
yt["label"] = yt["label"].map({0: "ham", 1: "spam"})
yt = yt[["text", "label", "source_video", "AUTHOR", "DATE"]]
yt.to_csv(os.path.join(OUT, "yt_spam.csv"), index=False)
print(f"[YT ] {n0_yt} -> {len(yt)} rows (bo {n0_yt - len(yt)} dup/null) | "
      f"labels: {yt['label'].value_counts().to_dict()}")

# ---------- 2) Vietnamese SMS spam ----------
vi = pd.read_csv(os.path.join(RAW_VI, "vi_dataset.csv"), encoding="utf-8-sig")
n0_vi = len(vi)
vi["labels"] = vi["labels"].astype(str).str.strip().str.lower()
vi["labels"] = vi["labels"].replace({"1": "spam"})  # nhan '1' = dac trung spam
vi = vi.rename(columns={"labels": "label", "texts_vi": "text"})
vi = vi.dropna(subset=["text"])
vi["text"] = vi["text"].astype(str).str.strip()
vi = vi[vi["text"] != ""]
vi = vi.drop_duplicates(subset=["text"]).reset_index(drop=True)
vi = vi[["text", "label"]]
vi.to_csv(os.path.join(OUT, "vi_spam.csv"), index=False)
print(f"[VI ] {n0_vi} -> {len(vi)} rows (bo {n0_vi - len(vi)} dup/null) | "
      f"labels: {vi['label'].value_counts().to_dict()}")
