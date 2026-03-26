
# FLAM R&D / AI Assignment Solution

## Problem Statement
We are tasked with estimating unknown parameters (θ, M, X) in the given parametric curve equations:

\[
x = (t \cos(\theta) - e^{M|t|} \sin(0.3t) \sin(\theta) + X) \\
y = (42 + t \sin(\theta) + e^{M|t|} \sin(0.3t) \cos(\theta))
\]

## Approach Overview
1. Loaded `xy_data.csv` containing points (x, y) sampled for t ∈ [6, 60].
2. Assumed data corresponds to uniformly spaced t values.
3. Implemented model equations in Python.
4. Defined **L1 loss** between observed and predicted points.
5. Used `scipy.optimize.differential_evolution` for global optimization and `minimize` for fine-tuning.

## Parameter Bounds
- θ ∈ (0°, 50°)
- M ∈ (-0.05, 0.05)
- X ∈ (0, 100)

## Results
| Parameter | Symbol | Estimated Value |
|------------|---------|----------------:|
| Theta (degrees) | θ | 28.12350 |
| M | M | 0.02142 |
| X | X | 54.89879 |

### Final Curve Equation
\[
\left(t * \cos(28.12350) - e^{0.02142|t|} \cdot \sin(0.3t) \sin(28.12350) + 54.89879,\ 
42 + t * \sin(28.12350) + e^{0.02142|t|} \cdot \sin(0.3t) \cos(28.12350) \right)
\]

This expression can be directly used in Desmos for verification.

## Files Included
- `fit_flam.py` — Python script to estimate parameters
- `xy_data.csv` — Provided data
- `fit_plot.png` — Visualization of fitted vs observed data
- `README.md` — Explanation of approach and results

## Author
Prepared by **Prakash Kumar Garg**



import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CONFIG
# =========================
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_saliency"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(pretrained=True)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# PROCESS FUNCTION
# =========================
def generate_saliency(image_path, save_path):
    # Load image
    img = Image.open(image_path).convert("RGB")
    input_img = transform(img).unsqueeze(0)

    # Enable gradient
    input_img.requires_grad_()

    # Forward pass
    output = model(input_img)
    pred_class = output.argmax(dim=1)

    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()

    # Get saliency
    saliency = input_img.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)

    saliency = saliency.squeeze().cpu().numpy()

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Convert to image
    saliency_img = (saliency * 255).astype(np.uint8)

    # Save using PIL
    saliency_pil = Image.fromarray(saliency_img)
    saliency_pil.save(save_path)

# =========================
# MAIN LOOP
# =========================
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"saliency_{filename}")

        print(f"Processing: {filename}")
        generate_saliency(input_path, output_path)

print("✅ Done! Saliency maps saved in:", OUTPUT_DIR)
