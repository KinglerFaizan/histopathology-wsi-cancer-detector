"""
src/wsi_tiling.py
-----------------
Whole-Slide Image (WSI) tiling engine.
Extracts non-overlapping 256×256 patches from gigapixel slides,
filters background using Otsu thresholding, and saves with spatial coordinates.

Requires: openslide-python
    pip install openslide-python
    (Also needs the OpenSlide C library: https://openslide.org/download/)
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logger.warning(
        "openslide-python not installed. WSI tiling unavailable.\n"
        "Install: pip install openslide-python\n"
        "Also install the C library: https://openslide.org/download/"
    )

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ──────────────────────────────────────────────
# Background / Tissue Detection
# ──────────────────────────────────────────────

def is_tissue_patch(patch: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Determine if a patch contains sufficient tissue.
    Uses Otsu thresholding on grayscale image.

    Args:
        patch     : (H, W, 3) RGB uint8 numpy array
        threshold : minimum fraction of non-background pixels

    Returns:
        True if tissue fraction >= threshold
    """
    gray = np.mean(patch, axis=2).astype(np.uint8)  # fast grayscale

    if CV2_AVAILABLE:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Fallback: simple intensity threshold
        otsu_val = np.mean(gray)
        binary = (gray < otsu_val).astype(np.uint8) * 255

    tissue_fraction = binary.sum() / (255 * binary.size)
    return tissue_fraction >= threshold


# ──────────────────────────────────────────────
# WSI Tiler
# ──────────────────────────────────────────────

class WSITiler:
    """
    Extracts patches from a Whole-Slide Image (WSI) file.

    Supported formats (via OpenSlide): .svs, .ndpi, .mrxs, .tiff, .scn

    Usage:
        tiler = WSITiler(slide_path='slide.svs', patch_size=256, level=0)
        patches = tiler.extract_patches(output_dir='./tiles/slide001/')
        print(f"Extracted {len(patches)} tissue patches")
    """

    def __init__(self, slide_path: str, patch_size: int = 256,
                 overlap: int = 0, level: int = 0,
                 tissue_threshold: float = 0.5):
        """
        Args:
            slide_path        : path to WSI file
            patch_size        : tile size in pixels (at target level)
            overlap           : overlap between adjacent tiles in pixels
            level             : OpenSlide pyramid level (0 = highest resolution)
            tissue_threshold  : min tissue fraction to keep tile (0–1)
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "openslide-python required for WSI tiling.\n"
                "pip install openslide-python"
            )

        self.slide_path = slide_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.level = level
        self.tissue_threshold = tissue_threshold
        self.slide_name = Path(slide_path).stem

        self.slide = openslide.OpenSlide(slide_path)
        self.width, self.height = self.slide.level_dimensions[level]
        self.downsample = self.slide.level_downsamples[level]

        logger.info(
            f"WSI loaded: {self.slide_name} | "
            f"Size: {self.width}×{self.height} px | "
            f"Level: {level} | Downsample: {self.downsample:.1f}×"
        )

    def get_thumbnail(self, max_size: int = 1024) -> np.ndarray:
        """Return a downscaled thumbnail for visualization."""
        thumb = self.slide.get_thumbnail((max_size, max_size))
        return np.array(thumb.convert('RGB'))

    def extract_patches(self, output_dir: str = None,
                        save_images: bool = True) -> list:
        """
        Tile the WSI and optionally save patches to disk.

        Returns:
            List of dicts: [{path, x, y, tissue_fraction, slide_name}, ...]
        """
        stride = self.patch_size - self.overlap
        scale = int(self.downsample)

        n_cols = (self.width  - self.overlap) // stride
        n_rows = (self.height - self.overlap) // stride
        total_tiles = n_rows * n_cols

        if output_dir and save_images:
            os.makedirs(output_dir, exist_ok=True)

        patches_info = []
        kept = 0

        logger.info(f"Tiling {self.slide_name}: {n_cols}×{n_rows} = {total_tiles:,} candidate tiles")

        for row in range(n_rows):
            for col in range(n_cols):
                # Coordinates at level-0 (full resolution)
                x0 = col * stride * scale
                y0 = row * stride * scale

                region = self.slide.read_region(
                    location=(x0, y0),
                    level=self.level,
                    size=(self.patch_size, self.patch_size)
                )
                patch = np.array(region.convert('RGB'))

                if not is_tissue_patch(patch, self.tissue_threshold):
                    continue

                kept += 1
                patch_name = f"{self.slide_name}_x{x0}_y{y0}.png"
                patch_path = os.path.join(output_dir, patch_name) if output_dir else None

                if patch_path and save_images:
                    Image.fromarray(patch).save(patch_path)

                patches_info.append({
                    'path': patch_path,
                    'x': x0, 'y': y0,
                    'col': col, 'row': row,
                    'slide_name': self.slide_name,
                })

        logger.info(
            f"Tiling complete: {kept:,}/{total_tiles:,} tissue patches "
            f"({kept/total_tiles:.1%} tissue coverage)"
        )

        # Save metadata
        if output_dir:
            meta_path = os.path.join(output_dir, 'patch_metadata.json')
            with open(meta_path, 'w') as f:
                json.dump(patches_info, f, indent=2)
            logger.info(f"Metadata saved to {meta_path}")

        return patches_info

    def __del__(self):
        if hasattr(self, 'slide'):
            self.slide.close()


# ──────────────────────────────────────────────
# Reconstruct Probability Map from Patch Scores
# ──────────────────────────────────────────────

def reconstruct_probability_map(patches_info: list,
                                 patch_scores: np.ndarray,
                                 slide_width: int,
                                 slide_height: int,
                                 patch_size: int = 256,
                                 downsample: int = 32) -> np.ndarray:
    """
    Reconstruct a spatial probability heatmap from patch-level scores.

    Args:
        patches_info : list of patch metadata dicts (from WSITiler)
        patch_scores : array of tumor probabilities per patch
        slide_width  : original slide width at level 0
        slide_height : original slide height at level 0
        patch_size   : size of each patch in pixels
        downsample   : factor to scale down the output map

    Returns:
        prob_map : (H//downsample, W//downsample) float32 numpy array
    """
    out_w = slide_width  // downsample
    out_h = slide_height // downsample
    prob_map = np.zeros((out_h, out_w), dtype=np.float32)
    count_map = np.zeros_like(prob_map)

    tile_size_out = max(1, patch_size // downsample)

    for info, score in zip(patches_info, patch_scores):
        x_out = info['x'] // downsample
        y_out = info['y'] // downsample
        x_end = min(x_out + tile_size_out, out_w)
        y_end = min(y_out + tile_size_out, out_h)

        prob_map[y_out:y_end, x_out:x_end] += score
        count_map[y_out:y_end, x_out:x_end] += 1

    # Average overlapping tiles
    prob_map = np.where(count_map > 0, prob_map / count_map, 0)
    return prob_map


# ──────────────────────────────────────────────
# Demo with synthetic slide
# ──────────────────────────────────────────────

def demo_tiling_synthetic(output_dir: str = '../outputs/demo_tiles',
                           n_patches: int = 16, patch_size: int = 256):
    """
    Simulate tiling output with synthetic patches (no real WSI needed).
    Useful for testing the downstream pipeline.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    patches_info = []
    for i in range(n_patches):
        patch = np.random.randint(180, 255, (patch_size, patch_size, 3), dtype=np.uint8)
        # Simulate H&E-like coloring
        patch[:, :, 0] = np.clip(patch[:, :, 0] - 30, 0, 255)
        patch[:, :, 2] = np.clip(patch[:, :, 2] - 20, 0, 255)

        row, col = divmod(i, 4)
        x0, y0 = col * patch_size, row * patch_size
        patch_name = f"demo_x{x0}_y{y0}.png"
        patch_path = os.path.join(output_dir, patch_name)
        Image.fromarray(patch).save(patch_path)

        patches_info.append({'path': patch_path, 'x': x0, 'y': y0,
                              'slide_name': 'demo', 'col': col, 'row': row})

    logger.info(f"Generated {n_patches} synthetic patches in {output_dir}")
    return patches_info


if __name__ == '__main__':
    patches = demo_tiling_synthetic()
    print(f"Demo tiles created: {len(patches)} patches")
    print(f"Sample: {patches[0]}")
