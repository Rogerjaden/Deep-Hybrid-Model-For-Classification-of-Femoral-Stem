
import torch
import os
import cv2
import numpy as np
from models.msftnet import MSFTNet

# 1. SETUP DUMMY DATA
print("Creating dummy dataset and test images...")
os.makedirs("dataset/anatomical", exist_ok=True)
os.makedirs("dataset/cemented", exist_ok=True)
os.makedirs("dataset/uncemented", exist_ok=True)
os.makedirs("test_images", exist_ok=True)

def create_random_image(path):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(path, img)

for i in range(2):
    create_random_image(f"dataset/anatomical/dummy_{i}.png")
    create_random_image(f"dataset/cemented/dummy_{i}.png")
    create_random_image(f"dataset/uncemented/dummy_{i}.png")
    create_random_image(f"test_images/sample_{i}.png")

# 2. GENERATE PLACEHOLDER MODEL
print("Generating placeholder model weight (msftnet_model.pth)...")
model = MSFTNet(num_classes=3)
torch.save(model.state_dict(), "msftnet_model.pth")

print("\nReady! You can now run:")
print("1. python predict.py  (to test prediction)")
print("2. python evaluate.py (to run evaluation)")
print("3. python main.py     (to see a single training step)")
