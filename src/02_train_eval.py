"""Bước 2 — Tiền xử lý, huấn luyện và đánh giá.

Pipeline cho từng bộ dữ liệu (yt_spam, vi_spam):
  1. Tách train/test stratified 80/20 (test giữ lại hoàn toàn, không đụng tới
     khi chọn mô hình — chống rò rỉ dữ liệu).
  2. Từ tập train: 5-fold cross-validation cho 5 mô hình (kịch bản "phương
     pháp tổng hợp kiến thức đã học" theo yêu cầu môn):
       - Naive Bayes (MultinomialNB)
       - Logistic Regression
       - SVM tuyến tính (LinearSVC)
       - K-Nearest Neighbors
       - Random Forest
     Trích xuất đặc trưng: Bag-of-Words và TF-IDF (so sánh 2 cách biểu diễn).
  3. Chọn mô hình tốt nhất theo CV F1, fit lại trên toàn bộ train, đánh giá
     trên test: Accuracy, Precision, Recall, F1 + confusion matrix.
  4. Phân tích lỗi: liệt kê mẫu test bị dự đoán sai.
  5. Lưu metrics (JSON), bảng so sánh (CSV), confusion matrix (PNG),
     model tốt nhất (.joblib).

Chạy: .venv/bin/python src/02_train_eval.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1_SPAM = make_scorer(f1_score, pos_label="spam")
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "outputs", "figures")
MET = os.path.join(ROOT, "outputs", "metrics")
MOD = os.path.join(ROOT, "outputs", "models")
for d in (FIG, MET, MOD):
    os.makedirs(d, exist_ok=True)

SEED = 42

def make_models(seed):
    return {
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "LinearSVM": LinearSVC(random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
    }

VECTORIZERS = {
    "BoW": lambda: CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95),
    "TFIDF": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True),
}

def evaluate(name, results_rows, best_by_ds, errors_by_ds):
    df = pd.read_csv(os.path.join(PROC, f"{name}.csv"))
    X, y = df["text"].astype(str), df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\n===== {name} =====")
    print(f"train={len(X_tr)}  test={len(X_te)}  "
          f"ty le spam: train={ (y_tr=='spam').mean():.1%}  test={(y_te=='spam').mean():.1%}")

    for vec_name, vec_factory in VECTORIZERS.items():
        for model_name, model in make_models(SEED).items():
            pipe = Pipeline([("vec", vec_factory()), ("clf", model)])
            scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring=F1_SPAM, n_jobs=-1)
            results_rows.append({
                "dataset": name, "features": vec_name, "model": model_name,
                "cv_f1_mean": round(scores.mean(), 4),
                "cv_f1_std": round(scores.std(), 4),
            })
            print(f"  CV  {vec_name:6} {model_name:20} F1={scores.mean():.4f} (+-{scores.std():.4f})")

    # Chọn cấu hình tốt nhất (vec + model) theo CV F1
    rows_ds = [r for r in results_rows if r["dataset"] == name]
    best = max(rows_ds, key=lambda r: r["cv_f1_mean"])
    vec_factory = VECTORIZERS[best["features"]]
    best_model = make_models(SEED)[best["model"]]

    # Tuning nhẹ tham số chính của mô hình tốt nhất (vẫn chỉ dùng train)
    param_grids = {
        "NaiveBayes": {"clf__alpha": [0.1, 0.5, 1.0]},
        "LogisticRegression": {"clf__C": [0.5, 1.0, 2.0]},
        "LinearSVM": {"clf__C": [0.5, 1.0, 2.0]},
        "KNN": {"clf__n_neighbors": [3, 5, 7]},
        "RandomForest": {"clf__n_estimators": [100, 200]},
    }
    pipe = Pipeline([("vec", vec_factory()), ("clf", best_model)])
    gs = GridSearchCV(pipe, param_grids[best["model"]], cv=5, scoring=F1_SPAM, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    print(f"  Best: {best['features']} + {best['model']} | tuning -> {gs.best_params_} "
          f"| CV F1={gs.best_score_:.4f}")

    # Đánh giá trên test (lần đầu chạm test)
    y_pred = gs.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, pos_label="spam")
    rec = recall_score(y_te, y_pred, pos_label="spam")
    f1 = f1_score(y_te, y_pred, pos_label="spam")
    print(f"  TEST: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

    metrics = {
        "dataset": name, "features": best["features"], "model": best["model"],
        "best_params": gs.best_params_, "cv_f1": round(gs.best_score_, 4),
        "test_accuracy": round(acc, 4), "test_precision": round(prec, 4),
        "test_recall": round(rec, 4), "test_f1": round(f1, 4),
        "train_size": len(X_tr), "test_size": len(X_te), "random_state": SEED,
    }
    best_by_ds[name] = metrics

    with open(os.path.join(MET, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Greens")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion matrix (test) — {name}\n{best['features']} + {best['model']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}_confusion_matrix.png"), dpi=130)
    plt.close(fig)

    # Phân tích lỗi: các mẫu dự đoán sai trên test
    err_idx = y_te.index[y_pred != y_te]
    errs = df.loc[err_idx, ["text", "label"]].copy()
    errs["pred"] = pd.Series(y_pred, index=X_te.index).loc[err_idx]
    errs.to_csv(os.path.join(MET, f"{name}_errors.csv"), index=False)
    errors_by_ds[name] = len(errs)
    print(f"  Sai {len(errs)}/{len(X_te)} mau test -> outputs/metrics/{name}_errors.csv")

    # Lưu model tốt nhất
    joblib.dump(gs.best_estimator_, os.path.join(MOD, f"{name}_best_model.joblib"))


if __name__ == "__main__":
    results_rows, best_by_ds, errors_by_ds = [], {}, {}
    for name in ("yt_spam", "vi_spam"):
        evaluate(name, results_rows, best_by_ds, errors_by_ds)

    cmp = pd.DataFrame(results_rows)
    cmp = cmp.sort_values(["dataset", "cv_f1_mean"], ascending=[True, False])
    cmp.to_csv(os.path.join(MET, "summary.csv"), index=False)
    with open(os.path.join(MET, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best_by_ds, "errors": errors_by_ds}, f,
                  ensure_ascii=False, indent=2)
    print("\n=== Bảng so sánh (top 5 mỗi dataset) ===")
    print(cmp.groupby("dataset").head(5).to_string(index=False))
