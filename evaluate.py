"""
src/evaluate.py
---------------
Comprehensive evaluation of the trained patch classifier:
  - AUC-ROC, F1, Accuracy, Sensitivity, Specificity
  - Confusion matrix
  - Optimal threshold search (Youden's J)
  - Per-class precision/recall
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader: DataLoader, device: str) -> tuple:
    """
    Run inference on a DataLoader.

    Returns:
        y_true  : np.ndarray of true labels
        y_prob  : np.ndarray of predicted probabilities
    """
    model.eval()
    all_labels, all_probs = [], []

    for batch in loader:
        if len(batch) == 2:
            imgs, labels = batch
        else:
            imgs = batch
            labels = None

        imgs = imgs.to(device)
        probs = torch.sigmoid(model(imgs)).squeeze().cpu().numpy()
        all_probs.extend(np.atleast_1d(probs).tolist())

        if labels is not None:
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


# ──────────────────────────────────────────────
# Threshold Selection
# ──────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                            method: str = 'youden') -> float:
    """
    Find optimal classification threshold.

    Args:
        method : 'youden' (maximise sensitivity+specificity) |
                 'f1' (maximise F1 score)
    """
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = thresholds[best_idx]
        logger.info(f"Youden threshold: {threshold:.4f} | "
                    f"Sensitivity: {tpr[best_idx]:.4f} | Specificity: {1-fpr[best_idx]:.4f}")

    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])
        threshold = thresholds[best_idx]
        logger.info(f"F1 threshold: {threshold:.4f} | F1: {f1_scores[best_idx]:.4f}")
    else:
        threshold = 0.5

    return float(threshold)


# ──────────────────────────────────────────────
# Full Evaluation
# ──────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_prob: np.ndarray,
             threshold: float = None) -> dict:
    """
    Compute all metrics.

    Args:
        y_true    : ground truth binary labels
        y_prob    : predicted probabilities
        threshold : if None, use Youden's J optimal threshold

    Returns:
        metrics dict
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob, method='youden')

    y_pred = (y_prob >= threshold).astype(int)

    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    f1   = f1_score(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)   # Recall / True Positive Rate
    specificity = tn / (tn + fp + 1e-8)   # True Negative Rate

    metrics = {
        'auc_roc':     round(auc, 4),
        'avg_precision': round(ap, 4),
        'f1_score':    round(f1, 4),
        'accuracy':    round(acc, 4),
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'threshold':   round(threshold, 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Tumor']))

    return metrics


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   save_path: str = None):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, color='steelblue', label=f'EfficientNet-B4 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.5000)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve — Histopathology Tumor Detector', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           save_path: str = None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ['d', '.1%'],
        ['Confusion Matrix (Counts)', 'Confusion Matrix (%)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['Normal', 'Tumor'],
                    yticklabels=['Normal', 'Tumor'],
                    ax=ax, linewidths=0.5)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(history: dict, save_path: str = None):
    """Plot train/val loss and AUC curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-o', ms=4, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   'r-o', ms=4, label='Val Loss')
    axes[0].set_title('Loss per Epoch', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_auc'], 'b-o', ms=4, label='Train AUC')
    axes[1].plot(epochs, history['val_auc'],   'r-o', ms=4, label='Val AUC')
    axes[1].axhline(y=max(history['val_auc']), color='gray', linestyle='--',
                    alpha=0.7, label=f"Best Val AUC: {max(history['val_auc']):.4f}")
    axes[1].set_title('AUC-ROC per Epoch', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Demo with synthetic predictions
    print("Running evaluation demo with synthetic predictions...")
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    # Simulate a decent model: higher probs for positives
    y_prob = np.where(y_true == 1,
                      np.random.beta(5, 2, n),
                      np.random.beta(2, 5, n))

    metrics = evaluate(y_true, y_prob)
    y_pred = (y_prob >= metrics['threshold']).astype(int)
    plot_roc_curve(y_true, y_prob, save_path='../outputs/roc_curve_demo.png')
    plot_confusion_matrix(y_true, y_pred, save_path='../outputs/confusion_matrix_demo.png')
    print("\nDemo evaluation complete ✅")
