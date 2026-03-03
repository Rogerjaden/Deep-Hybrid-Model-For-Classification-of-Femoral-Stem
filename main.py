import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import logging
from datetime import datetime

from models.msftnet import MSFTNet
from utils.dataset import HipDataset
from utils.transforms import get_train_transforms, get_valid_transforms
from utils.engine import train_one_epoch, validate

# -----------------------------
# CONFIG & LOGGING
# -----------------------------
DATASET_PATH = "dataset"
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4

# Create Logs directory
os.makedirs("Logs", exist_ok=True)

# Set up logging
log_filename = f"Logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
logger.info(f"Using device: {device}")

# -----------------------------
# DATASET
# -----------------------------
full_dataset = HipDataset(
    DATASET_PATH,
    transform=get_train_transforms()
)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size

train_ds, valid_ds = random_split(full_dataset, [train_size, valid_size])

valid_ds.dataset.transform = get_valid_transforms()

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

# -----------------------------
# MODEL
# -----------------------------
num_classes = len(full_dataset.class_to_idx)

model = MSFTNet(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.cuda.amp.GradScaler()

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, scaler, device, criterion
    )

    val_loss, val_acc = validate(
        model, valid_loader, device, criterion
    )

    scheduler.step()

    logger.info(f"""
Epoch {epoch+1}/{EPOCHS}
Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}
Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f}
""")
    torch.save(model.state_dict(), "msftnet_model.pth")
    logger.info("Model saved!")

