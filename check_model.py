import torch
try:
    data = torch.load("msftnet_model.pth", map_location="cpu")
    print("Keys in checkpoint:", data.keys() if isinstance(data, dict) else "Data is not a dict")
    print("Type of data:", type(data))
except Exception as e:
    print("Error loading model:", e)
