"""
Step 2 (start): Track vehicles across a video and build a count-over-time dataset.

Detection alone tells you "3 cars in this frame." Tracking assigns each vehicle
a persistent ID across frames, so we can tell how many DISTINCT vehicles passed
through, and count traffic density per second -- the input our LSTM predictor
will eventually train on.

Usage:
    python src/detection/track_vehicles.py MVI_20011
    (pass any UA-DETRAC sequence name that exists under data/raw/ua-detrac/DETRAC-Images/DETRAC-Images/)
"""
import sys
import csv
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO

MODEL_PATH = "models/yolov8n_uadetrac_v1.pt"
IMAGES_ROOT = Path("data/raw/ua-detrac/DETRAC-Images/DETRAC-Images")
VIDEO_CACHE_DIR = Path("outputs/videos")
COUNTS_OUTPUT_DIR = Path("data/processed/counts")
OUTPUT_FPS = 10  # frames per second for the video we build (source frames are ~25fps;
                  # we don't need full framerate for counting purposes)

CLASS_NAMES = ["car", "bus", "van", "others"]


def build_video_from_frames(sequence_dir: Path, out_path: Path, fps: int = OUTPUT_FPS):
    """Stitch a sequence's frames into an mp4 so YOLO's tracker can process it as a video."""
    frame_paths = sorted(sequence_dir.glob("img*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frames found in {sequence_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    h, w = first_frame.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in frame_paths:
        frame = cv2.imread(str(p))
        writer.write(frame)
    writer.release()
    return len(frame_paths)


def track_and_count(video_path: Path, sequence_name: str):
    """Run tracking across the video, return per-second counts and total unique vehicles."""
    model = YOLO(MODEL_PATH)

    seen_ids = set()
    # time_bin (int seconds) -> set of track_ids active in that bin
    bin_to_ids = defaultdict(set)

    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        conf=0.3,
        save=True,
        project="outputs",
        name=f"track_{sequence_name}",
        stream=True,  # process frame-by-frame instead of loading everything into memory
        verbose=False,
    )

    for frame_idx, r in enumerate(results):
        time_sec = frame_idx // OUTPUT_FPS
        if r.boxes.id is None:
            continue
        for track_id, cls_id in zip(r.boxes.id.tolist(), r.boxes.cls.tolist()):
            track_id = int(track_id)
            seen_ids.add(track_id)
            bin_to_ids[time_sec].add(track_id)

    return seen_ids, bin_to_ids


def write_counts_csv(bin_to_ids: dict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    max_bin = max(bin_to_ids.keys()) if bin_to_ids else 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "vehicle_count"])
        for sec in range(max_bin + 1):
            writer.writerow([sec, len(bin_to_ids.get(sec, set()))])


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/detection/track_vehicles.py <SEQUENCE_NAME>  e.g. MVI_20011")

    sequence_name = sys.argv[1]
    sequence_dir = IMAGES_ROOT / sequence_name
    if not sequence_dir.exists():
        raise SystemExit(f"Sequence not found: {sequence_dir}")

    video_path = VIDEO_CACHE_DIR / f"{sequence_name}.mp4"
    if not video_path.exists():
        print(f"Building video from frames in {sequence_dir} ...")
        n_frames = build_video_from_frames(sequence_dir, video_path)
        print(f"  wrote {video_path} ({n_frames} frames)")
    else:
        print(f"Using cached video: {video_path}")

    print("Running tracking ...")
    seen_ids, bin_to_ids = track_and_count(video_path, sequence_name)

    out_csv = COUNTS_OUTPUT_DIR / f"{sequence_name}_counts.csv"
    write_counts_csv(bin_to_ids, out_csv)

    print("\nDone.")
    print(f"  Total distinct vehicles tracked: {len(seen_ids)}")
    print(f"  Per-second counts written to: {out_csv}")
    print(f"  Annotated video saved under: outputs/track_{sequence_name}/")


if __name__ == "__main__":
    main()
