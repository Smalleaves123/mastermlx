from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mastermlx.linear_models import LogisticRegression
from mastermlx.neighbors import KNNClassifier
from mastermlx.preprocessing import StandardScaler as LocalStandardScaler
from mastermlx.probabilistic import GaussianNB
from mastermlx.trees import DecisionTreeClassifier
from mastermlx.viz import plot_confusion_matrix, plot_interactive_scatter, plot_loss_curve
from mastermlx.decomposition import PCA


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    accuracy = float(np.mean(y_true == y_pred))
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def fit_and_score(name, model, X_train, X_test, y_train, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start

    metrics = classification_metrics(y_test, y_pred)
    metrics.update({
        "model": name,
        "fit_time_sec": fit_time,
        "predict_time_sec": predict_time,
    })
    return metrics, model, y_pred


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(path, rows):
    names = [row["model"] for row in rows]
    accuracy = [row["accuracy"] for row in rows]
    fit_time = [row["fit_time_sec"] for row in rows]
    predict_time = [row["predict_time_sec"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, accuracy, color="#4C72B0")
    axes[0].set_title("Test accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(names, fit_time, label="fit", color="#55A868")
    axes[1].bar(names, predict_time, bottom=fit_time, label="predict", color="#C44E52")
    axes[1].set_title("Runtime")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrices(output_dir, y_test, predictions, labels):
    for name, y_pred in predictions.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_confusion_matrix(y_test, y_pred, labels=labels, ax=ax, title=f"{name} confusion matrix")
        fig.savefig(output_dir / f"iris_confusion_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close(fig)


def save_pca_plot(output_dir, X, y, labels):
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    fig = plot_interactive_scatter(Z, y, title="Iris PCA projection")
    fig.write_html(str(output_dir / "iris_pca_projection.html"))


def main():
    try:
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier as SkKNNClassifier
        from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB as SkGaussianNB
    except ImportError as exc:
        raise SystemExit("scikit-learn is required to run this benchmark") from exc

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target
    labels = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = LocalStandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    models = [
        ("Logistic Regression", LogisticRegression(lr=0.1, n_iter=2000, random_state=0)),
        ("KNN", KNNClassifier(k=5)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=4)),
        ("GaussianNB", GaussianNB()),
        ("Sklearn Logistic Regression", SkLogisticRegression(max_iter=1000)),
        ("Sklearn KNN", SkKNNClassifier(n_neighbors=5)),
        ("Sklearn Decision Tree", SkDecisionTreeClassifier(max_depth=4, random_state=0)),
        ("Sklearn GaussianNB", SkGaussianNB()),
    ]

    rows = []
    predictions = {}
    fitted_models = {}
    for name, model in models:
        metrics, fitted, y_pred = fit_and_score(name, model, X_train, X_test, y_train, y_test)
        rows.append(metrics)
        predictions[name] = y_pred
        fitted_models[name] = fitted

    fieldnames = ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "fit_time_sec", "predict_time_sec"]
    save_csv(output_dir / "iris_classification_results.csv", rows, fieldnames)
    save_summary_plot(output_dir / "iris_classification_summary.png", rows)
    save_confusion_matrices(output_dir, y_test, predictions, labels)
    save_pca_plot(output_dir, X_test, y_test, labels)

    log_model = fitted_models["Logistic Regression"]
    if getattr(log_model, "loss_", None):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_loss_curve(log_model.loss_, ax=ax, title="Logistic Regression training loss")
        fig.savefig(output_dir / "iris_logistic_loss.png", dpi=160)
        plt.close(fig)

    rows_sorted = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
    print("Iris classification benchmark")
    print("============================")
    for row in rows_sorted:
        print(
            f"{row['model']:<28} "
            f"acc={row['accuracy']:.3f} "
            f"f1={row['f1_macro']:.3f} "
            f"fit={row['fit_time_sec']:.4f}s "
            f"pred={row['predict_time_sec']:.4f}s"
        )
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
