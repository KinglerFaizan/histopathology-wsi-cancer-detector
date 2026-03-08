"""
src/slide_inference.py
----------------------
Patch-to-slide inference pipeline.
Aggregates patch-level probabilities to a slide-level tumor prediction
using max pooling, mean-of-top-K, and attention-weighted aggregation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
import os

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Aggregation Methods
# ──────────────────────────────────────────────

def aggregate_max(patch_probs: np.ndarray) -> float:
    """Slide score = max patch probability. Very sensitive."""
    return float(patch_probs.max())


def aggregate_mean_topk(patch_probs: np.ndarray, k: int = 10) -> float:
    """Slide score = mean of top-K patch probabilities."""
    topk = np.sort(patch_probs)[::-1][:k]
    return float(topk.mean())


def aggregate_threshold_ratio(patch_probs: np.ndarray,
                               threshold: float = 0.5) -> float:
    """Slide score = fraction of patches above threshold."""
    return float((patch_probs >= threshold).mean())


# ──────────────────────────────────────────────
# Attention-based MIL Aggregation
# ──────────────────────────────────────────────

class AttentionAggregator(nn.Module):
    """
    Attention-based Multiple Instance Learning (MIL) aggregation.

    Learns to weight patches by their importance.
    Reference: Ilse et al., "Attention-based Deep MIL", ICML 2018.

    Input : bag of patch feature vectors (N_patches, feature_dim)
    Output: slide-level prediction (scalar) + attention weights (N_patches,)
    """

    def __init__(self, feature_dim: int = 1792, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor):
        """
        Args:
            features : (N_patches, feature_dim)

        Returns:
            logit           : scalar prediction
            attention_weights : (N_patches,) softmax weights
        """
        attn_raw = self.attention(features)                   # (N, 1)
        attn_weights = F.softmax(attn_raw, dim=0)             # (N, 1)
        slide_repr = (attn_weights * features).sum(dim=0)     # (feature_dim,)
        logit = self.classifier(slide_repr)
        return logit.squeeze(), attn_weights.squeeze()


# ──────────────────────────────────────────────
# Full Slide Inference
# ──────────────────────────────────────────────

@torch.no_grad()
def run_slide_inference(model, patches_info: list, transform,
                         device: str, batch_size: int = 64,
                         aggregation: str = 'mean_topk',
                         top_k: int = 10) -> dict:
    """
    Run full slide inference: extract patch features → aggregate → predict.

    Args:
        model        : trained HistoClassifier
        patches_info : list of patch metadata (from WSITiler)
        transform    : inference albumentations transform
        device       : 'cuda' or 'cpu'
        batch_size   : inference batch size
        aggregation  : 'max' | 'mean_topk' | 'threshold_ratio'
        top_k        : k for mean_topk aggregation

    Returns:
        dict with slide_score, patch_probs, aggregation_method
    """
    from PIL import Image
    import numpy as np

    model.eval()
    patch_probs = []

    # Batch inference
    for start in range(0, len(patches_info), batch_size):
        batch_info = patches_info[start:start + batch_size]
        imgs = []
        for info in batch_info:
            img_np = np.array(Image.open(info['path']).convert('RGB'))
            t = transform(image=img_np)['image']
            imgs.append(t)

        batch_tensor = torch.stack(imgs).to(device)
        probs = torch.sigmoid(model(batch_tensor)).squeeze().cpu().numpy()
        patch_probs.extend(np.atleast_1d(probs).tolist())

    patch_probs = np.array(patch_probs)

    # Aggregate
    if aggregation == 'max':
        slide_score = aggregate_max(patch_probs)
    elif aggregation == 'mean_topk':
        slide_score = aggregate_mean_topk(patch_probs, k=top_k)
    elif aggregation == 'threshold_ratio':
        slide_score = aggregate_threshold_ratio(patch_probs)
    else:
        slide_score = float(patch_probs.mean())

    logger.info(f"Slide inference complete: {len(patch_probs)} patches | "
                f"Slide score ({aggregation}): {slide_score:.4f}")

    return {
        'slide_score': slide_score,
        'patch_probs': patch_probs,
        'aggregation': aggregation,
        'n_patches': len(patch_probs),
        'n_tumor_patches': int((patch_probs >= 0.5).sum()),
    }


# ──────────────────────────────────────────────
# Spatial Heatmap
# ──────────────────────────────────────────────

def plot_spatial_heatmap(patches_info: list, patch_probs: np.ndarray,
                          patch_size: int = 256, save_path: str = None):
    """
    Reconstruct and plot a 2D spatial probability heatmap of the slide.
    """
    if not patches_info:
        logger.warning("No patch info provided for heatmap.")
        return

    max_col = max(p['col'] for p in patches_info) + 1
    max_row = max(p['row'] for p in patches_info) + 1

    grid = np.zeros((max_row, max_col))
    for info, prob in zip(patches_info, patch_probs):
        grid[info['row'], info['col']] = prob

    fig, ax = plt.subplots(figsize=(max(6, max_col // 4), max(5, max_row // 4)))
    im = ax.imshow(grid, cmap='hot_r', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Tumor Probability')
    ax.set_title('Slide-Level Tumor Probability Heatmap', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tile Column')
    ax.set_ylabel('Tile Row')

    # Annotate high-probability regions
    tumor_coords = [(p['row'], p['col']) for p, prob in zip(patches_info, patch_probs)
                    if prob >= 0.8]
    if tumor_coords:
        rows, cols = zip(*tumor_coords)
        ax.scatter(cols, rows, c='cyan', s=20, marker='x', alpha=0.6, label='High prob (≥0.8)')
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Heatmap saved → {save_path}")
    plt.show()


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("Slide inference demo with synthetic patch scores...")
    np.random.seed(42)

    n_patches = 64
    patches_info = [{'col': i % 8, 'row': i // 8, 'path': None}
                    for i in range(n_patches)]

    # Simulate a tumor region in the top-left quadrant
    patch_probs = np.random.beta(2, 5, n_patches)
    for i, p in enumerate(patches_info):
        if p['row'] < 4 and p['col'] < 4:
            patch_probs[i] = np.random.beta(5, 2)

    os.makedirs('../outputs', exist_ok=True)
    print(f"Max score: {aggregate_max(patch_probs):.4f}")
    print(f"Top-10 mean: {aggregate_mean_topk(patch_probs, 10):.4f}")

    plot_spatial_heatmap(patches_info, patch_probs,
                         save_path='../outputs/spatial_heatmap_demo.png')
    print("Demo complete ✅")
