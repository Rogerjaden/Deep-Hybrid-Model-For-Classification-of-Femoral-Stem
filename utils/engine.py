import torch
from tqdm import tqdm
from utils.metrics import accuracy

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    """
    Core training loop for a single epoch.
    Includes Mixed Precision (AMP) support for NVIDIA GPUs.
    """
    model.train()
    total_loss = 0
    total_acc = 0
    
    # Progress bar for feedback
    pbar = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision (AMP) logic: Accelerates training and saves memory
        if device == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            # Scaled backpropagation to prevent underflow in FP16
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training fallback
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        acc = accuracy(outputs, labels).item()
        total_acc += acc
        
        # Live status update
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    return total_loss/len(loader), total_acc/len(loader)


def validate(model, loader, device, criterion):
    """
    Inference-only validation loop to evaluate current model state.
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Use AMP during inference for consistency with training
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            acc = accuracy(outputs, labels).item()
            total_acc += acc
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    return total_loss/len(loader), total_acc/len(loader)
