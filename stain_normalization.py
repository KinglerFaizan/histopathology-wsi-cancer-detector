"""
src/stain_normalization.py
--------------------------
Stain normalization for H&E histopathology images.
Implements:
  1. Macenko method  — SVD-based optical density stain separation
  2. Reinhard method — Color statistics transfer in LAB color space
"""

import numpy as np
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# Shared Utilities
# ──────────────────────────────────────────────

def rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Convert RGB image (uint8) to Optical Density (OD) space."""
    img = img.astype(np.float64)
    img = np.clip(img, 1, 255)
    return -np.log(img / 255.0)


def od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Convert Optical Density back to RGB uint8."""
    rgb = np.exp(-od) * 255
    return np.clip(rgb, 0, 255).astype(np.uint8)


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """L2-normalize each row of matrix A."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (norms + 1e-8)


# ──────────────────────────────────────────────
# Macenko Stain Normalization
# ──────────────────────────────────────────────

class MacenkoNormalizer:
    """
    Macenko et al. (2009) stain normalization.
    Decomposes H&E image into stain matrix via SVD in OD space.

    Usage:
        normalizer = MacenkoNormalizer()
        normalizer.fit(target_image)          # target slide reference
        norm_img = normalizer.transform(source_image)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        self.alpha = alpha   # percentile for robust SVD
        self.beta = beta     # OD threshold for tissue mask
        self.stain_matrix_target = None
        self.maxC_target = None

    def fit(self, target: np.ndarray):
        """Estimate stain matrix from a target (reference) image."""
        self.stain_matrix_target, self.maxC_target = self._get_stain_matrix(target)

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Normalize source image to match target stain."""
        if self.stain_matrix_target is None:
            raise RuntimeError("Call fit() with a target image first.")

        stain_matrix_source, maxC_source = self._get_stain_matrix(source)

        # Get concentration of each stain in source
        source_od = rgb_to_od(source).reshape(-1, 3)
        C_source = np.linalg.lstsq(stain_matrix_source.T, source_od.T, rcond=None)[0].T

        # Normalize concentrations
        C_norm = C_source * (self.maxC_target / (maxC_source + 1e-8))

        # Reconstruct in OD space using target stain matrix
        norm_od = C_norm @ self.stain_matrix_target
        norm_rgb = od_to_rgb(norm_od.reshape(source.shape))
        return norm_rgb

    def _get_stain_matrix(self, img: np.ndarray):
        """Extract stain matrix and max concentrations from image."""
        od = rgb_to_od(img).reshape(-1, 3)

        # Remove background pixels (low OD)
        od_hat = od[np.all(od > self.beta, axis=1)]

        if len(od_hat) < 10:
            # Fallback: use all pixels
            od_hat = od

        # SVD to find principal stain directions
        _, _, Vt = np.linalg.svd(od_hat, full_matrices=False)
        plane = Vt[:2].T  # Project onto 2D plane

        # Project to unit sphere and find angular extremes
        proj = od_hat @ plane
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)

        v1 = np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = np.array([np.cos(max_phi), np.sin(max_phi)])

        stain1 = plane @ v1
        stain2 = plane @ v2

        # Sort: haematoxylin first (higher blue channel in OD)
        if stain1[0] < stain2[0]:
            stain_matrix = np.array([stain1, stain2])
        else:
            stain_matrix = np.array([stain2, stain1])

        stain_matrix = normalize_rows(stain_matrix)

        # Get max stain concentrations (percentile for robustness)
        C = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
        maxC = np.percentile(C, 99, axis=0)

        return stain_matrix, maxC


# ──────────────────────────────────────────────
# Reinhard Stain Normalization
# ──────────────────────────────────────────────

class ReinhardNormalizer:
    """
    Reinhard et al. color transfer in LAB color space.
    Transfers mean and std of each LAB channel from target to source.

    Simpler and faster than Macenko; works well for inter-scanner variation.
    """

    def __init__(self):
        self.target_mean = None
        self.target_std = None

    def fit(self, target: np.ndarray):
        """Compute LAB statistics of the target image."""
        lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        self.target_mean = lab.mean(axis=(0, 1))
        self.target_std = lab.std(axis=(0, 1))

    def transform(self, source: np.ndarray) -> np.ndarray:
        """Transfer target color statistics to source."""
        if self.target_mean is None:
            raise RuntimeError("Call fit() with a target image first.")

        lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean = lab.mean(axis=(0, 1))
        src_std  = lab.std(axis=(0, 1))

        # Normalize: subtract source stats, scale to target stats
        norm_lab = (lab - src_mean) / (src_std + 1e-8) * self.target_std + self.target_mean
        norm_lab = np.clip(norm_lab, 0, 255).astype(np.uint8)

        result = cv2.cvtColor(norm_lab, cv2.COLOR_LAB2RGB)
        return result


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def get_normalizer(method: str = 'macenko'):
    """
    Get a stain normalizer by name.

    Args:
        method: 'macenko' | 'reinhard' | 'none'
    """
    if method == 'macenko':
        return MacenkoNormalizer()
    elif method == 'reinhard':
        return ReinhardNormalizer()
    elif method == 'none':
        return None
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print("Stain normalization demo with synthetic patches...")

    # Create synthetic H&E-like patches (pink/purple tones)
    np.random.seed(42)
    source = np.random.randint(180, 255, (96, 96, 3), dtype=np.uint8)
    source[:, :, 0] = np.clip(source[:, :, 0] - 30, 0, 255)  # reduce red → purplish

    target = np.random.randint(160, 240, (96, 96, 3), dtype=np.uint8)
    target[:, :, 2] = np.clip(target[:, :, 2] - 40, 0, 255)  # reduce blue → pinkish

    # Reinhard (more stable on synthetic)
    r_norm = ReinhardNormalizer()
    r_norm.fit(target)
    norm_img = r_norm.transform(source)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [source, target, norm_img],
                               ['Source', 'Target', 'Reinhard Normalized']):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('../outputs/stain_normalization_demo.png', dpi=120, bbox_inches='tight')
    plt.show()
    print("Demo saved to outputs/stain_normalization_demo.png ✅")
