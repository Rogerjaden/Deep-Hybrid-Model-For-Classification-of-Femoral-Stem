import torch
from utils.metrics import accuracy

def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()

    total_loss = 0
    total_acc = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(outputs, labels).item()

    return total_loss/len(loader), total_acc/len(loader)


def validate(model, loader, device, criterion):
    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_acc += accuracy(outputs, labels).item()

    return total_loss/len(loader), total_acc/len(loader)
