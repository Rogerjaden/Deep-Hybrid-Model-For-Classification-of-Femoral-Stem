import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)

        # Hook directly on saved feature map
        self.model.feature_map.register_hook(self.save_gradient)

        score = output[:, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.model.feature_map

        weights = gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam