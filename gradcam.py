"""
src/gradcam.py
--------------
Grad-CAM (Gradient-weighted Class Activation Mapping) for
histopathology patch explainability.

Highlights the regions within a patch that the model uses
to predict tumor presence — critical for pathologist review.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Hook Manager
# ──────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation compatible with EfficientNet and ResNet backbones.

    Usage:
        gcam = GradCAM(model, target_layer_name='encoder.features.8')
        heatmap = gcam.generate(img_tensor)
        overlay = gcam.overlay(original_image, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None
        self._hooks = []

        # Auto-detect target layer if not specified
        if target_layer_name is None:
            target_layer_name = self._auto_detect_layer()

        target_layer = self._get_layer(target_layer_name)
        self._register_hooks(target_layer)
        logger.info(f"Grad-CAM initialized. Target layer: {target_layer_name}")

    def _auto_detect_layer(self) -> str:
        """Auto-detect the last convolutional layer."""
        name = self.model.backbone_name

        if 'efficientnet' in name:
            return 'encoder.blocks'   # Last conv block in EfficientNet
        elif 'resnet' in name:
            return 'encoder.6'        # layer4 in ResNet (index 6 in Sequential)
        else:
            # Fallback: find last Conv2d
            last_conv_name = None
            for n, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    last_conv_name = n
            return last_conv_name

    def _get_layer(self, name: str) -> nn.Module:
        """Retrieve a submodule by dotted name."""
        parts = name.split('.')
        module = self.model
        for p in parts:
            if p.isdigit():
                module = module[int(p)]
            else:
                module = getattr(module, p)
        return module

    def _register_hooks(self, layer: nn.Module):
        """Register forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = layer.register_forward_hook(forward_hook)
        h2 = layer.register_full_backward_hook(backward_hook)
        self._hooks = [h1, h2]

    def generate(self, img_tensor: torch.Tensor,
                  target_class: int = 1) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.

        Args:
            img_tensor   : (1, 3, H, W) normalized image tensor
            target_class : class index to explain (1 = tumor)

        Returns:
            heatmap : (H, W) float32 numpy array in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(img_tensor)
        score = logits[0, 0]  # Binary: single logit

        # Backward pass to get gradients
        score.backward()

        # Grad-CAM computation
        grads = self.gradients          # (1, C, H', W')
        acts  = self.activations        # (1, C, H', W')

        # Global average pooling of gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)                                 # keep only positive influence

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.astype(np.float32)

    def overlay(self, original_img: np.ndarray, heatmap: np.ndarray,
                alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_img : (H, W, 3) uint8 RGB numpy array
            heatmap      : (H', W') float32 numpy array in [0, 1]
            alpha        : heatmap opacity (0 = invisible, 1 = opaque)
            colormap     : OpenCV colormap (COLORMAP_JET, COLORMAP_HOT, etc.)

        Returns:
            overlay : (H, W, 3) uint8 RGB numpy array
        """
        h, w = original_img.shape[:2]

        # Resize heatmap to input resolution
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Blend with original
        overlay = (alpha * colored_rgb + (1 - alpha) * original_img).astype(np.uint8)
        return overlay

    def visualize(self, original_img: np.ndarray, img_tensor: torch.Tensor,
                   prob: float = None, save_path: str = None,
                   patch_id: str = ''):
        """
        Full visualization: original | heatmap | overlay.
        """
        heatmap = self.generate(img_tensor)
        overlay_img = self.overlay(original_img, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        titles = ['Original Patch', 'Grad-CAM Heatmap', 'Overlay']
        imgs   = [original_img,
                  plt.cm.jet(heatmap)[:, :, :3],   # matplotlib colormap
                  overlay_img]

        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

        prob_str = f" | Tumor Prob: {prob:.3f}" if prob is not None else ''
        fig.suptitle(f"Grad-CAM Explainability{prob_str} | {patch_id}",
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Grad-CAM saved → {save_path}")
        plt.show()

    def remove_hooks(self):
        """Remove all registered hooks (cleanup)."""
        for h in self._hooks:
            h.remove()


# ──────────────────────────────────────────────
# Batch Grad-CAM
# ──────────────────────────────────────────────

def batch_gradcam(model: nn.Module, patch_paths: list, transform,
                   device: str, output_dir: str,
                   top_n: int = 20, prob_threshold: float = 0.5):
    """
    Generate Grad-CAM heatmaps for the top-N highest-probability patches.

    Args:
        model          : trained HistoClassifier
        patch_paths    : list of image file paths
        transform      : inference transform (albumentations)
        device         : 'cuda' or 'cpu'
        output_dir     : directory to save heatmap images
        top_n          : number of patches to visualize
        prob_threshold : minimum probability to include
    """
    import os
    from PIL import Image as PILImage
    os.makedirs(output_dir, exist_ok=True)

    gcam = GradCAM(model)
    model.eval()
    results = []

    # Score all patches first
    logger.info(f"Scoring {len(patch_paths)} patches...")
    with torch.no_grad():
        for path in patch_paths:
            img_np = np.array(PILImage.open(path).convert('RGB'))
            t = transform(image=img_np)['image'].unsqueeze(0).to(device)
            prob = torch.sigmoid(model(t)).item()
            results.append((prob, path, img_np, t))

    # Sort by probability descending
    results.sort(key=lambda x: x[0], reverse=True)
    high_prob = [(p, path, img_np, t) for p, path, img_np, t in results
                 if p >= prob_threshold][:top_n]

    logger.info(f"Generating Grad-CAM for {len(high_prob)} high-probability patches...")
    for i, (prob, path, img_np, t) in enumerate(high_prob):
        patch_id = f"patch_{i+1:03d}"
        save_path = os.path.join(output_dir, f"gradcam_{patch_id}.png")
        gcam.visualize(img_np, t, prob=prob,
                       save_path=save_path, patch_id=patch_id)

    gcam.remove_hooks()
    logger.info(f"✅ Grad-CAM complete. {len(high_prob)} heatmaps saved to {output_dir}")


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from model import HistoClassifier

    print("Grad-CAM demo with random weights (no real model needed)...")

    model = HistoClassifier(backbone='resnet50', pretrained=False)
    model.eval()

    # Synthetic patch
    dummy_np  = np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)
    dummy_np[:100, :100, 0] = 80   # simulate a "dark" region

    dummy_tensor = torch.randn(1, 3, 224, 224)

    gcam = GradCAM(model, target_layer_name='encoder.6')
    heatmap = gcam.generate(dummy_tensor)
    overlay = gcam.overlay(dummy_np, heatmap)

    print(f"Heatmap shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    print(f"Overlay shape: {overlay.shape}")

    os.makedirs('../outputs', exist_ok=True)
    gcam.visualize(dummy_np, dummy_tensor, prob=0.78,
                   save_path='../outputs/gradcam_demo.png',
                   patch_id='demo_patch')
    gcam.remove_hooks()
    print("Grad-CAM demo complete ✅")
