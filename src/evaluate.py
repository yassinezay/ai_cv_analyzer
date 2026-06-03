"""
evaluate.py – Metrics, confusion matrix, and training curve visualisations
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Inference ─────────────────────────────────────────────────────────────────

def get_predictions(model, loader, device: str = "cpu"):
    """Run model on a DataLoader and return (predictions, true_labels) arrays."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, class_names=None, verbose: bool = True) -> dict:
    """Compute accuracy, F1, precision, recall. Print report if verbose."""
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted",    zero_division=0),
    }
    if verbose:
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1-score  : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        if class_names is not None:
            print("\n" + classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0
            ))
    return metrics


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, title: str = "Confusion Matrix"):
    """Heatmap confusion matrix with seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,             fontsize=9)
    plt.tight_layout()
    plt.show()
    return cm


def plot_training_curves(history: dict, title: str = "Training History"):
    """Side-by-side loss and accuracy curves for a single experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", ms=4, label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", ms=4, label="Validation")
    axes[0].set(title=f"{title} – Loss",     xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", ms=4, label="Train")
    axes[1].plot(epochs, history["val_acc"],   "r-o", ms=4, label="Validation")
    axes[1].set(title=f"{title} – Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_experiments_comparison(experiments: list):
    """Overlay val-loss and val-accuracy for multiple hyperparameter runs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for exp in experiments:
        name = exp["config"]["name"]
        h    = exp["history"]
        ep   = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(ep, h["val_loss"], "-o", ms=4, label=name)
        axes[1].plot(ep, h["val_acc"],  "-o", ms=4, label=name)

    titles  = ["Val Loss – Hyperparameter Comparison",
               "Val Accuracy – Hyperparameter Comparison"]
    ylabels = ["Loss", "Accuracy"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set(title=t, xlabel="Epoch", ylabel=yl)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_overfitting(history: dict, threshold: float = 0.08):
    """Diagnose overfitting from a training history dict."""
    tr_l = history["train_loss"][-1]
    vl_l = history["val_loss"][-1]
    tr_a = history["train_acc"][-1]
    vl_a = history["val_acc"][-1]
    gap_l = vl_l - tr_l
    gap_a = tr_a  - vl_a

    print("═" * 52)
    print("  Overfitting Analysis")
    print("═" * 52)
    print(f"  Train Loss : {tr_l:.4f}  |  Val Loss : {vl_l:.4f}  Δ={gap_l:+.4f}")
    print(f"  Train Acc  : {tr_a:.4f}  |  Val Acc  : {vl_a:.4f}  Δ={gap_a:+.4f}")
    if gap_l > threshold or gap_a > threshold:
        print("\n  ⚠  Overfitting detected.")
        print("     Applied mitigations: Dropout, L2 weight-decay,")
        print("     gradient clipping (norm=1), LR step-decay scheduler.")
    else:
        print("\n  ✓  No significant overfitting – model generalises well.")
    print("═" * 52)
    return gap_l, gap_a


def experiments_table(experiments: list, baseline_metrics: dict = None):
    """Print a formatted table of all training runs."""
    sep = "─" * 90
    print("\n" + "═" * 90)
    print("  EXPERIMENTS TABLE")
    print("═" * 90)
    print(f"{'#':<4}{'Model':<32}{'LR':<9}{'Hidden':<9}{'Dropout':<10}"
          f"{'Val Acc':<11}{'F1':<9}{'Time(s)'}")
    print(sep)

    if baseline_metrics:
        print(f"{'B':<4}{'TF-IDF + LogReg':<32}{'–':<9}{'–':<9}{'–':<10}"
              f"{baseline_metrics['accuracy']:<11.4f}"
              f"{baseline_metrics['f1']:<9.4f}"
              f"{baseline_metrics['train_time']:.1f}")

    for i, exp in enumerate(experiments, 1):
        c = exp["config"]
        h = exp["history"]
        m = exp.get("metrics", {})
        print(f"{i:<4}{c['name']:<32}{c['lr']:<9}"
              f"{c.get('hidden_dim','–'):<9}{c.get('dropout','–'):<10}"
              f"{h['val_acc'][-1]:<11.4f}"
              f"{m.get('f1', 0):<9.4f}"
              f"{h['train_time']:.1f}")

    print("═" * 90)
