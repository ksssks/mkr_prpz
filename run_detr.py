import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrImageProcessor, DetrForObjectDetection

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Шляхи до вхідної та вихідної папок
input_dir = "detr_images"
output_dir = "runs/detr_predict"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_dir):
    raise FileNotFoundError(f"Папку '{input_dir}' не знайдено в корені проєкту.")

# Отримуємо перші 10 зображень із папки detr_images
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
image_files = sorted([
    f for f in os.listdir(input_dir)
    if f.lower().endswith(valid_extensions)
])[:10]

if not image_files:
    raise FileNotFoundError(f"У папці '{input_dir}' не знайдено жодного зображення.")

# 2. Завантаження попередньо навченої моделі DETR (на базі ResNet-50)
print("Завантаження моделі DETR...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

num_images = len(image_files)

# 3. Обробка зображень через DETR
for i, img_name in enumerate(image_files, start=1):
    img_path = os.path.join(input_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    # Підготовка зображення та інференс
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Постобробка результатів (поріг впевненості 0.7)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.7
    )[0]

    print(f"\n[{i}/{num_images}] Результати для {img_name}:")
    if len(results["scores"]) == 0:
        print("  - Об'єкти з порогом впевненості > 0.7 не знайдено")

    # Візуалізація та збереження
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(coord, 2) for coord in box.tolist()]
        class_name = model.config.id2label[label.item()]
        print(f"  - Знайдено: {class_name} (впевненість: {round(score.item(), 3)}) | Координати: {box}")

        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            xmin, ymin - 5, f"{class_name}: {round(score.item(), 2)}",
            color="white", fontsize=10, bbox=dict(facecolor="red", alpha=0.7, pad=2)
        )

    plt.axis("off")
    save_path = os.path.join(output_dir, f"detr_{img_name}")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

print(f"\nГотово! Оброблені зображення збережено у '{output_dir}'.")