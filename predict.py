import torch
import cv2
import numpy as np
import os
from models.msftnet import MSFTNet
from gradcam import GradCAM

MODEL_PATH = "msftnet_model.pth"
TEST_FOLDER = "test_images"
OUTPUT_FOLDER = "results/gradcam"

CLASS_NAMES = ["anatomical", "cemented", "uncemented"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = MSFTNet(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Target last CNN layer (ResNet final block)
target_layer = model.backbone.layer4[-1]
gradcam = GradCAM(model)

for filename in os.listdir(TEST_FOLDER):

    image_path = os.path.join(TEST_FOLDER, filename)
    print(f"\nProcessing: {filename}")

    orig = cv2.imread(image_path)
    if orig is None:
        print("Could not read image")
        continue

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160,160))

    img = img / 255.0
    img = (img - 0.5) / 0.5

    img_tensor = np.transpose(img, (2,0,1))
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    print("Predicted:", CLASS_NAMES[pred_class])
    print("Confidence:", probs[0][pred_class].item())

    # Grad-CAM
    cam = gradcam.generate(img_tensor, pred_class)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

    # Overlay heatmap
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # Resize both for display
    display_size = (400, 400)
    raw_display = cv2.resize(orig, display_size)
    overlay_display = cv2.resize(overlay, display_size)

    # Add titles
    cv2.putText(raw_display, "Sample Input (Raw X-Ray)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.putText(overlay_display, "Model Explanation (Grad-CAM)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    # Add prediction text
    confidence = probs[0][pred_class].item()
    label_text = f"Prediction: {CLASS_NAMES[pred_class]} ({confidence:.2f})"

    cv2.putText(overlay_display, label_text,
                (20, 370), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    # Combine side-by-side
    combined = np.hstack((raw_display, overlay_display))

    # Save combined image
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, combined)

    # Show popup window
    cv2.imshow("Grad-CAM Visualization", combined)
    cv2.waitKey(0)  # Press any key to move to next image
    cv2.destroyAllWindows()