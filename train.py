"""
train.py
--------
Entry point for training the histopathology patch classifier.

Usage:
    python train.py
    python train.py --config configs/config.yaml --model resnet50
    python train.py --config configs/config.yaml --model efficientnet_b4 --epochs 30
"""

import argparse
import yaml
import os
import torch
import logging
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Histopathology Classifier')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default=None,
                        help='Override backbone: efficientnet_b4 | resnet50')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cuda | cpu (auto-detected by default)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model:
        config['model']['backbone'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device.upper()}")

    # Create output dirs
    for dir_key in config.get('paths', {}).values():
        os.makedirs(dir_key, exist_ok=True)

    # Build DataLoaders
    logger.info("Building DataLoaders...")
    try:
        from dataset import get_dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir=config['data']['data_dir'],
            img_size=config['data'].get('img_size', 224),
            batch_size=config['training']['batch_size'],
            num_workers=config['data'].get('num_workers', 4),
        )
    except FileNotFoundError as e:
        logger.error(f"\n{'='*60}")
        logger.error("Dataset not found!")
        logger.error(str(e))
        logger.error("\nPlease follow the instructions in data/README.md to download PCam.")
        logger.error(f"{'='*60}")
        return

    # Train
    from train import train as run_training
    model, history = run_training(config, train_loader, val_loader, device=device)

    # Evaluate on test set
    logger.info("\n--- Test Set Evaluation ---")
    from evaluate import run_inference, evaluate, plot_roc_curve, plot_confusion_matrix, plot_training_history

    y_true, y_prob = run_inference(model, test_loader, device)
    metrics = evaluate(y_true, y_prob)

    # Save plots
    results_dir = config['paths']['results_dir']
    plot_roc_curve(y_true, y_prob,
                   save_path=os.path.join(results_dir, 'roc_curve.png'))
    plot_training_history(history,
                          save_path=os.path.join(results_dir, 'training_history.png'))

    y_pred = (y_prob >= metrics['threshold']).astype(int)
    plot_confusion_matrix(y_true, y_pred,
                          save_path=os.path.join(results_dir, 'confusion_matrix.png'))

    logger.info(f"\n🎯 Final Test Results:")
    logger.info(f"   AUC-ROC:     {metrics['auc_roc']}")
    logger.info(f"   F1 Score:    {metrics['f1_score']}")
    logger.info(f"   Sensitivity: {metrics['sensitivity']}")
    logger.info(f"   Specificity: {metrics['specificity']}")


if __name__ == '__main__':
    main()
