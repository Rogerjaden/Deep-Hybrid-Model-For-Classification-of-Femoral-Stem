import os
import cv2
from torch.utils.data import Dataset

class HipDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls:i for i,cls in enumerate(classes)}

        for cls in classes:
            path = os.path.join(root_dir, cls)
            for img in os.listdir(path):
                self.images.append(os.path.join(path,img))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, self.labels[idx]
