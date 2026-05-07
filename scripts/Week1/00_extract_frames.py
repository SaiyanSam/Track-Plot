"""
Usage:
python scripts/Week1/00_extract_frames.py \
  --video data/demo2.mp4 \
  --output data/frames \
  --stride 1

After running above, process using nerfstudio:
ns-process-data images \
  --data data/frames \
  --output-dir results/colmap_output \
  --matching-method sequential
  
After colmap, do the 3DGS:
ns-train nerfacto --data results/colmap_output
"""


import argparse
from pathlib import Path
import cv2


def extract_frames(video_path: Path, output_dir: Path, stride: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Saving every {stride} frame(s) to: {output_dir}")

    saved = 0 
    frame_idx = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_idx % stride == 0:
            out_path = output_dir / f"frame_{saved:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()

    print(f"Saved {saved} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/frames",
        help="Output frame directory",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Save every Nth frame",
    )

    args = parser.parse_args()

    extract_frames(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        stride=args.stride,
    )
