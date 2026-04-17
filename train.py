from ultralytics import YOLO
import yaml, shutil
from pathlib import Path

# ── Merge both datasets into one data.yaml ────────────────────
# plates_global  → train/val primary
# plates_lowlight → extra training images

# Fix paths in data.yaml to absolute
import os
base = "/coderepo/KP"

data = {
    "path": base,
    "train": [
        "datasets/plates_global/train/images",
        "datasets/plates_lowlight/train/images",
    ],
    "val": "datasets/plates_global/valid/images",
    "nc": 1,
    "names": ["license_plate"],
}

with open("combined_data.yaml", "w") as f:
    yaml.dump(data, f)

print("Starting training...")

model = YOLO("yolov8n.pt")
model.train(
    data="combined_data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,          # safe for Jetson shared memory
    device=0,
    project="uae_plate_model",
    name="v1",
    augment=True,
    hsv_h=0.02,       # slight hue shift (UAE plates vary by emirate)
    hsv_v=0.5,        # brightness aug — helps with night footage
    fliplr=0.0,       # don't flip plates horizontally
    mosaic=1.0,
    patience=15,      # early stop if no improvement
)

print("Done — best model at: uae_plate_model/v1/weights/best.pt")
