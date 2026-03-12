import os
import cv2
import numpy as np

def generate_images(target_dir, count, prefix="generated"):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Check for existing files with the same prefix to avoid overwriting
    existing_files = [f for f in os.listdir(target_dir) if f.startswith(prefix) and f.endswith(".png")]
    start_idx = len(existing_files)
    
    print(f"Generating {count} more images in {target_dir} (starting from index {start_idx})...")
    for i in range(start_idx, start_idx + count):
        # Generate a random grayscale image resembling an X-ray (simple noise/gradients)
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        # Add some shapes to make it less like pure noise
        cv2.circle(img, (112, 112), 50, (100, 100, 100), -1)
        
        filename = f"{prefix}_{i}.png"
        path = os.path.join(target_dir, filename)
        cv2.imwrite(path, img)

if __name__ == "__main__":
    base_path = r"D:\Malcolm\Craft\Projects\Deep-Hybrid-Model-For-Classification-of-Femoral-Stem\Deep-Hybrid-Model-For-Classification-of-Femoral-Stem-main\dataset"
    categories = ["anatomical", "cemented", "uncemented"]
    
    for cat in categories:
        target = os.path.join(base_path, cat)
        generate_images(target, 100, prefix=f"aug_{cat}")
    
    print("\nDataset generation complete.")
