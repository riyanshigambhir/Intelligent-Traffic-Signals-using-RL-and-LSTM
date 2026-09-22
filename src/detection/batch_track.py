"""
Step 2 (batch): Run tracking across multiple sequences to build a bigger
count-over-time dataset for the LSTM.

Runs track_vehicles.py's logic across N sequences automatically instead of
calling it one at a time. Also merges everything into one combined CSV with
a sequence column, which is what the LSTM prep step will want.

Usage:
    python src/detection/batch_track.py            # runs on 10 sequences (default)
    python src/detection/batch_track.py 20          # runs on 20 sequences
"""
import sys
import csv
from pathlib import Path

from track_vehicles import (
    IMAGES_ROOT, VIDEO_CACHE_DIR, COUNTS_OUTPUT_DIR,
    build_video_from_frames, track_and_count, write_counts_csv,
)

DEFAULT_N_SEQUENCES = 10
COMBINED_OUTPUT = Path("data/processed/counts/all_sequences_counts.csv")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_SEQUENCES

    all_sequences = sorted(p.name for p in IMAGES_ROOT.iterdir() if p.is_dir())
    if not all_sequences:
        raise SystemExit(f"No sequences found under {IMAGES_ROOT}")

    sequences = all_sequences[:n]
    print(f"Running tracking on {len(sequences)} sequences: {sequences}\n")

    combined_rows = []
    summary = []

    for i, seq_name in enumerate(sequences, 1):
        print(f"[{i}/{len(sequences)}] {seq_name} ...")
        sequence_dir = IMAGES_ROOT / seq_name

        video_path = VIDEO_CACHE_DIR / f"{seq_name}.mp4"
        if not video_path.exists():
            build_video_from_frames(sequence_dir, video_path)

        try:
            seen_ids, bin_to_ids = track_and_count(video_path, seq_name)
        except Exception as e:
            print(f"  [FAILED] {seq_name}: {e}")
            continue

        out_csv = COUNTS_OUTPUT_DIR / f"{seq_name}_counts.csv"
        write_counts_csv(bin_to_ids, out_csv)

        max_bin = max(bin_to_ids.keys()) if bin_to_ids else 0
        for sec in range(max_bin + 1):
            combined_rows.append([seq_name, sec, len(bin_to_ids.get(sec, set()))])

        summary.append((seq_name, len(seen_ids), max_bin + 1))
        print(f"  -> {len(seen_ids)} distinct vehicles, {max_bin + 1} seconds of data")

    COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "time_seconds", "vehicle_count"])
        writer.writerows(combined_rows)

    print("\n=== Summary ===")
    for seq_name, n_vehicles, n_secs in summary:
        print(f"  {seq_name}: {n_vehicles} vehicles, {n_secs}s")
    print(f"\nCombined dataset written to: {COMBINED_OUTPUT}")
    print(f"Total rows: {len(combined_rows)}")


if __name__ == "__main__":
    main()
