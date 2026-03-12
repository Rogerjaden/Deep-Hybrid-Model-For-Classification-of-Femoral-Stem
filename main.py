import os
import torch
import torch.nn as nn
import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from models.msftnet import MSFTNet
from utils.dataset import HipDataset
from utils.transforms import get_train_transforms, get_valid_transforms
from utils.engine import train_one_epoch, validate

# Disable symlinks warning for HuggingFace to keep logs clean
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def parse_args():
    """
    Parses command line arguments for training configuration.
    Allows easy experimentation without modifying source code.
    """
    parser = argparse.ArgumentParser(description="MSFT-Net Training Orchestrator")
    parser.add_argument("--data_path", type=str, default="dataset", help="Path to raw image dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Samples per gradient update")
    parser.add_argument("--epochs", type=int, default=50, help="Max training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for AdamW")
    parser.add_argument("--attn", type=str, default="eca", choices=["eca", "cbam", "none"], 
                        help="Select attention mechanism (ECA recommended for speed/params)")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Early stopping patience (epochs without improvement)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure necessary directories exist for artifacts
    os.makedirs("Logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Set up dual logging (File + Console) for persistence and real-time monitoring
    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"Logs/training_{log_time}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution Device: {device}")
    logger.info(f"Environment Configuration: {vars(args)}")

    # -----------------------------
    # DATASET & SPLITTING
    # -----------------------------
    # Instantiate dataset twice with different transforms to prevent augmentation leakage into validation
    full_dataset_train = HipDataset(args.data_path, transform=get_train_transforms())
    full_dataset_valid = HipDataset(args.data_path, transform=get_valid_transforms())

    # Stratified splitting: Ensures class distribution is uniform across splits
    indices = list(range(len(full_dataset_train)))
    train_indices, valid_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=full_dataset_train.labels
    )

    train_ds = Subset(full_dataset_train, train_indices)
    valid_ds = Subset(full_dataset_valid, valid_indices)

    # Optimized DataLoaders with pin_memory and multi-processing
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # -----------------------------
    # MODEL, LOSS, OPTIMIZER
    # -----------------------------
    num_classes = len(full_dataset_train.class_to_idx)
    model = MSFTNet(num_classes=num_classes, attn_type=args.attn).to(device)

    # CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # AdamW with weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine Annealing: Systematically reduces LR for smoother convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # GradScaler for Mixed Precision (FP16) training compatibility
    scaler = torch.cuda.amp.GradScaler()

    # -----------------------------
    # TRAIN LOOP (Enhanced with Best Model Tracking & Early Stopping)
    # -----------------------------
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_path = "msftnet_model_best.pth"

    for epoch in range(args.epochs):
        # Training Phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion
        )

        # Validation Phase
        val_loss, val_acc = validate(
            model, valid_loader, device, criterion
        )

        # Update Learning Rate
        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Checkpointing: Save only the best weights based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"!!! New Best Model Captured (Accuracy: {val_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Persistence: Save the latest state as a fallback
        torch.save(model.state_dict(), "msftnet_model_latest.pth")

        # Early Stopping check
        if epochs_no_improve >= args.patience:
            logger.warning(f"No improvement for {args.patience} epochs. Terminating training at Epoch {epoch+1}.")
            break

    logger.info(f"Optimization Finished. Peak Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
