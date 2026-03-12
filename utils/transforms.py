import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=224):
    """
    Returns robust image augmentations for training.
    Optimized for medical imaging (X-ray) by including Flips and Brightness adjustments.
    """
    return A.Compose([
        # Medical images are often variable in size; resize to consistent square input
        A.Resize(image_size, image_size),
        
        # Spatial Augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2), # Some X-rays might be inverted or rotated
        A.RandomRotate90(p=0.2),
        
        # Intensity Augmentations
        A.RandomBrightnessContrast(p=0.3),
        
        # Geometric distortions to simulate different positioning
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        
        # Noise simulation
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.Blur(blur_limit=3),
        ], p=0.2),
        
        # Standard ImageNet normalization (compatible with pretrained ResNet backbone)
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_valid_transforms(image_size=224):
    """
    Returns minimal transforms for validation/inference.
    Ensures data consistency without stochastic variations.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
