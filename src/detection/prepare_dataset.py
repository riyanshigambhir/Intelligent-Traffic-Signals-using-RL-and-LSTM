"""
Step 1b: Convert UA-DETRAC (images + XML annotations) into YOLOv8 training format.

UA-DETRAC gives us:
  - Images: data/raw/ua-detrac/DETRAC-Images/DETRAC-Images/<SEQUENCE>/img#####.jpg
  - Annotations: one XML per sequence, with <frame num="N"><target_list><target><box .../></target>

YOLOv8 wants:
  - dataset/images/train/<name>.jpg   + dataset/labels/train/<name>.txt (one label file per image)
  - dataset/images/val/<name>.jpg     + dataset/labels/val/<name>.txt
  - each label line: <class_id> <x_center> <y_center> <width> <height>   (all normalized 0-1)

We subsample every FRAME_STEP-th frame per sequence (full dataset is 140K images —
way more than a laptop needs to get a working model), and hold out the last
VAL_SEQUENCE_FRACTION of sequences entirely for validation (so we're testing on
sequences the model never saw at all, not just held-out frames from the same video).
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
from PIL import Image

# ---- Paths (adjust these two if your folder names differ) ----
IMAGES_ROOT = Path("data/raw/ua-detrac/DETRAC-Images/DETRAC-Images")
ANNOTATIONS_ROOT = Path("data/raw/ua-detrac/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML")
OUTPUT_ROOT = Path("data/processed/dataset")

# ---- Settings ----
FRAME_STEP = 10          # keep every 10th frame per sequence
VAL_SEQUENCE_FRACTION = 0.15   # ~15% of sequences held out entirely for validation
SEED = 42

CLASS_NAMES = ["car", "bus", "van", "others"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def convert_box(left, top, width, height, img_w, img_h):
    """Pixel box (left, top, width, height) -> YOLO normalized (x_center, y_center, w, h)."""
    x_center = (left + width / 2) / img_w
    y_center = (top + height / 2) / img_h
    return x_center, y_center, width / img_w, height / img_h


def process_sequence(xml_path: Path, images_dir: Path, out_images: Path, out_labels: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = root.get("name")

    written = 0
    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))
        if frame_num % FRAME_STEP != 0:
            continue

        img_name = f"img{frame_num:05d}.jpg"
        src_img = images_dir / img_name
        if not src_img.exists():
            continue

        with Image.open(src_img) as im:
            img_w, img_h = im.size

        lines = []
        target_list = frame.find("target_list")
        if target_list is not None:
            for target in target_list.findall("target"):
                box = target.find("box")
                attr = target.find("attribute")
                if box is None or attr is None:
                    continue
                vtype = attr.get("vehicle_type", "others")
                cls_id = CLASS_TO_ID.get(vtype, CLASS_TO_ID["others"])
                xc, yc, w, h = convert_box(
                    float(box.get("left")), float(box.get("top")),
                    float(box.get("width")), float(box.get("height")),
                    img_w, img_h,
                )
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        out_name = f"{seq_name}_{img_name}"
        shutil.copy(src_img, out_images / out_name)
        (out_labels / out_name.replace(".jpg", ".txt")).write_text("\n".join(lines))
        written += 1

    return written


def main():
    random.seed(SEED)
    xml_files = sorted(ANNOTATIONS_ROOT.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found in {ANNOTATIONS_ROOT} — check the path.")

    random.shuffle(xml_files)
    n_val = max(1, int(len(xml_files) * VAL_SEQUENCE_FRACTION))
    val_xmls = set(xml_files[:n_val])

    for split in ("train", "val"):
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_written = {"train": 0, "val": 0}
    for i, xml_path in enumerate(xml_files, 1):
        seq_name = xml_path.stem
        images_dir = IMAGES_ROOT / seq_name
        if not images_dir.exists():
            print(f"  [skip] no image folder for {seq_name}")
            continue

        split = "val" if xml_path in val_xmls else "train"
        out_images = OUTPUT_ROOT / "images" / split
        out_labels = OUTPUT_ROOT / "labels" / split

        written = process_sequence(xml_path, images_dir, out_images, out_labels)
        total_written[split] += written
        print(f"[{i}/{len(xml_files)}] {seq_name} -> {split} ({written} frames)")

    print("\nDone.")
    print(f"  train images: {total_written['train']}")
    print(f"  val images:   {total_written['val']}")

    # Write the YAML config YOLOv8 needs
    yaml_path = Path("data/processed/ua_detrac.yaml")
    yaml_path.write_text(
        f"path: {OUTPUT_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    print(f"  wrote config: {yaml_path}")


if __name__ == "__main__":
    main()
