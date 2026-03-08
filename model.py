"""
src/model.py
------------
Patch-level classifier for histopathology images.
Supports EfficientNet-B4 (primary) and ResNet-50 (baseline).
Uses pretrained ImageNet weights with a custom classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Classification Head
# ──────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Custom classification head:
    GAP → Dropout → FC(hidden_dim) → ReLU → Dropout → FC(1)
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ──────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────

class HistoClassifier(nn.Module):
    """
    Patch-level tumor classifier for histopathology images.

    Args:
        backbone   : model name ('efficientnet_b4' | 'resnet50' | 'resnet34')
        pretrained : use ImageNet pretrained weights
        hidden_dim : size of the hidden FC layer
        dropout    : dropout rate
        freeze_bn  : freeze BatchNorm layers (useful with small batches)
    """

    SUPPORTED_BACKBONES = ['efficientnet_b4', 'resnet50', 'resnet34', 'resnet18']

    def __init__(self, backbone: str = 'efficientnet_b4',
                 pretrained: bool = True,
                 hidden_dim: int = 256,
                 dropout: float = 0.4,
                 freeze_bn: bool = False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == 'efficientnet_b4':
            self.encoder = timm.create_model('efficientnet_b4', pretrained=pretrained,
                                              num_classes=0, global_pool='avg')
            in_features = self.encoder.num_features  # 1792 for B4

        elif backbone in ['resnet50', 'resnet34', 'resnet18']:
            weights = 'IMAGENET1K_V1' if pretrained else None
            _model = getattr(models, backbone)(weights=weights)
            in_features = _model.fc.in_features
            # Remove the original FC head
            self.encoder = nn.Sequential(*list(_model.children())[:-1],
                                          nn.Flatten())
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Choose from {self.SUPPORTED_BACKBONES}")

        self.classifier = ClassificationHead(in_features, hidden_dim, dropout)

        if freeze_bn:
            self._freeze_batchnorm()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model: {backbone} | Total params: {total_params/1e6:.1f}M | "
                    f"Trainable: {trainable_params/1e6:.1f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : (B, 3, H, W) input tensor

        Returns:
            logits : (B, 1) raw logits (no sigmoid)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (before classification head)."""
        return self.encoder(x)

    def freeze_encoder(self):
        """Freeze all encoder parameters (for fine-tuning head only)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen. Only classification head will be trained.")

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen. Full model will be trained.")

    def _freeze_batchnorm(self):
        """Freeze BatchNorm layers (set to eval mode)."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, '_freeze_bn_flag') and self._freeze_bn_flag:
            self._freeze_batchnorm()
        return self


# ──────────────────────────────────────────────
# Loss Function
# ──────────────────────────────────────────────

def get_loss_fn(pos_weight: float = 1.0, device: str = 'cpu') -> nn.Module:
    """
    Binary Cross-Entropy with Logits + optional positive class weighting.
    pos_weight > 1.0 penalizes false negatives more (useful for imbalanced data).
    """
    pw = torch.tensor([pos_weight], device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


# ──────────────────────────────────────────────
# Optimizer & Scheduler
# ──────────────────────────────────────────────

def get_optimizer(model: nn.Module, lr: float = 1e-4,
                  weight_decay: float = 1e-4, optimizer: str = 'adamw'):
    """Build optimizer with differential learning rates (encoder vs. head)."""
    head_params = list(model.classifier.parameters())
    head_ids = set(id(p) for p in head_params)
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]

    param_groups = [
        {'params': encoder_params, 'lr': lr * 0.1},    # slower for pretrained encoder
        {'params': head_params,    'lr': lr},            # faster for new head
    ]

    if optimizer == 'adamw':
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer == 'adam':
        return torch.optim.Adam(param_groups)
    elif optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def get_scheduler(optimizer, scheduler: str = 'cosine',
                  epochs: int = 30, warmup_epochs: int = 2):
    """Build LR scheduler."""
    if scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
        )
    elif scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")


# ──────────────────────────────────────────────
# Checkpoint Utilities
# ──────────────────────────────────────────────

def save_checkpoint(model: nn.Module, optimizer, epoch: int,
                     val_auc: float, path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc,
        'backbone': model.backbone_name,
    }, path)
    logger.info(f"Checkpoint saved → {path} (epoch={epoch}, AUC={val_auc:.4f})")


def load_checkpoint(model: nn.Module, path: str,
                     optimizer=None, device: str = 'cpu') -> dict:
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    logger.info(f"Checkpoint loaded ← {path} (epoch={ckpt.get('epoch')}, "
                f"AUC={ckpt.get('val_auc', '?'):.4f})")
    return ckpt


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    for backbone in ['resnet50', 'efficientnet_b4']:
        try:
            model = HistoClassifier(backbone=backbone, pretrained=False).to(device)
            dummy = torch.randn(4, 3, 224, 224).to(device)
            out = model(dummy)
            proba = model.predict_proba(dummy)
            print(f"✅ {backbone}: input {dummy.shape} → logits {out.shape} → proba {proba.shape}")
        except Exception as e:
            print(f"⚠️  {backbone}: {e}")
