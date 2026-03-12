import os
import cv2
import numpy as np
import random

def augment_image(image):
    # Random Rotation
    angle = random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random Flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    # Random Brightness/Contrast
    alpha = random.uniform(0.8, 1.2) # Contrast
    beta = random.uniform(-20, 20)   # Brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Random Noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
    return image

def generate_dataset(base_dir, categories, target_count=100):
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"Directory {cat_dir} does not exist. Skipping.")
            continue
        
        # 1. Clean up old noise-based 'aug_' images
        files = os.listdir(cat_dir)
        for f in files:
            if f.startswith("aug_"):
                os.remove(os.path.join(cat_dir, f))
        
        # 2. Find source images (Samples)
        source_images = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith("sample_aug_")]
        
        if not source_images:
            print(f"No source images found in {cat_dir}. Cannot augment.")
            continue
            
        print(f"Found {len(source_images)} source images in {cat}. Generating {target_count} augmentations...")
        
        for i in range(target_count):
            source_file = random.choice(source_images)
            img = cv2.imread(os.path.join(cat_dir, source_file))
            if img is None:
                continue
                
            aug_img = augment_image(img)
            
            output_name = f"sample_aug_{i}.png"
            cv2.imwrite(os.path.join(cat_dir, output_name), aug_img)

if __name__ == "__main__":
    base_path = r"D:\Malcolm\Craft\Projects\Deep-Hybrid-Model-For-Classification-of-Femoral-Stem\Deep-Hybrid-Model-For-Classification-of-Femoral-Stem-main\dataset"
    categories = ["anatomical", "cemented", "uncemented"]
    generate_dataset(base_path, categories, 100)
    print("\nSmart dataset augmentation complete.")
