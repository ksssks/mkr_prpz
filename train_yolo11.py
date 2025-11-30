from ultralytics import YOLO
import os

root = "augmented_dataset"
required_dirs = [
    "images/train", "images/val", "images/test",
    "labels/train", "labels/val", "labels/test",
]

print("\n Checking dataset structure...")
for d in required_dirs:
    path = os.path.join(root, d)
    if not os.path.exists(path):
        raise FileNotFoundError(f" Missing folder: {path}")
    else:
        print(f" {path}")

print("\nStructure OK!\n")

print(" Loading YOLOv11 model...")
model = YOLO("yolo11n.pt")

print(" Starting training...\n")
model.train(
    data="augmented_dataset.yaml",
    epochs=15,
    imgsz=512,
    batch=4,
    workers=2,
    cache=True,
    save=True,
    pretrained=True,
    device="cpu"
)

print("\n Training completed!")

print("\n Validating model...")
metrics = model.val()

print("\nMetrics:", metrics)

print("\n Running predictions on test set...")
model.predict(
    source="augmented_dataset/images/test",
    save=True,
    conf=0.25
)

print("\n Done! Check the 'runs/detect' folder.")
