import os
import cv2
import albumentations as A


# ============ CONFIG ============
INPUT_DIR = "dataset"
OUTPUT_DIR = "augmented_dataset"
AUG_PER_IMAGE = 5


# ============ TRANSFORMS ============
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.4),
    A.Blur(blur_limit=3, p=0.2),
    A.CLAHE(p=0.2),
    A.RandomGamma(p=0.3),

    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(0.05, 0.05),
        rotate=(-20, 20),
        shear=(-10, 10),
        p=0.7
    ),

    A.CoarseDropout(
        max_holes=4,
        max_height=40,
        max_width=40,
        p=0.3
    ),

    A.RGBShift(p=0.3)
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2)
)


# ============ FUNCTIONS ============
def load_yolo_labels(label_path):
    boxes = []
    classes = []

    if not os.path.exists(label_path):
        return boxes, classes

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())

            if w == 0 or h == 0:
                continue

            boxes.append([x, y, w, h])
            classes.append(int(cls))

    return boxes, classes


def save_yolo_labels(output_path, bboxes, classes):
    with open(output_path, "w") as f:
        for cls, (x, y, w, h) in zip(classes, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def process_split(split):
    img_dir = f"{INPUT_DIR}/images/{split}"
    lbl_dir = f"{INPUT_DIR}/labels/{split}"

    out_img_dir = f"{OUTPUT_DIR}/images/{split}"
    out_lbl_dir = f"{OUTPUT_DIR}/labels/{split}"

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for filename in os.listdir(img_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, filename)
        lbl_path = os.path.join(lbl_dir, filename.rsplit(".", 1)[0] + ".txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"Не можу прочитати фото: {img_path}")
            continue

        bboxes, classes = load_yolo_labels(lbl_path)

        if len(bboxes) == 0:
            print(f"⚠ Пропускаю (немає боксів): {filename}")
            continue

        for i in range(AUG_PER_IMAGE):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            except Exception as e:
                print(f"Пропуск через помилку bbox ({filename}): {e}")
                continue

            if len(augmented["bboxes"]) == 0:
                print(f"⚠ Bbox зник після аугментації: {filename}")
                continue

            new_img_name = filename.rsplit(".", 1)[0] + f"_aug{i}.jpg"
            new_lbl_name = filename.rsplit(".", 1)[0] + f"_aug{i}.txt"

            cv2.imwrite(os.path.join(out_img_dir, new_img_name), augmented["image"])
            save_yolo_labels(os.path.join(out_lbl_dir, new_lbl_name),
                             augmented["bboxes"], augmented["class_labels"])


# ================= RUN ==================
for split in ("train", "val", "test"):
    print(f"▶ Обробляю {split}...")
    process_split(split)

print("\nГотово! Дані збережено в:", OUTPUT_DIR)
