import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader
from src.tracking.detector import YOLOTracker
from src.geometry.projection import (
    estimate_ground_y,
    pixel_to_ground_ray,
    bbox_bottom_center,
)


def main():
    transforms_path = PROJECT_ROOT / "results/colmap_output/transforms.json"
    images_dir = PROJECT_ROOT / "results/colmap_output/images"
    output_csv = PROJECT_ROOT / "results/trajectory_raw.csv"

    loader = DataLoader(transforms_path)

    K = loader.get_intrinsics()
    extrinsics = loader.get_extrinsics()
    image_names = loader.get_image_paths()

    cx = K[0, 2]
    cy = K[1, 2]

    ground_y = estimate_ground_y(
        K=K,
        extrinsics=extrinsics,
        cx=cx,
        cy=cy,
    )

    print(f"Estimated ground Y: {ground_y:.6f}")

    tracker = YOLOTracker(
        model_name="yolov8x.pt",
        classes=[2, 7],
        confidence=0.1,
        image_size=1280,
        tracker="botsort.yaml",
    )

    target_id = None
    rows = []

    for frame_idx, image_name in enumerate(image_names):
        image_path = images_dir / image_name

        if image_name not in extrinsics:
            continue

        detections = tracker.track_image_path(image_path)

        if len(detections) == 0:
            continue

        if target_id is None:
            # choose highest-confidence vehicle in first valid frame
            detections = sorted(
                detections,
                key=lambda d: d["confidence"],
                reverse=True,
            )
            target_id = detections[0]["track_id"]
            print(f"Selected target ID: {target_id}")

        target_det = None

        for det in detections:
            if det["track_id"] == target_id:
                target_det = det
                break

        if target_det is None:
            continue

        u, v = bbox_bottom_center(target_det["bbox_xywh"])

        point_world = pixel_to_ground_ray(
            K=K,
            c2w=extrinsics[image_name],
            u=u,
            v=v,
            ground_y=ground_y,
        )

        if point_world is None:
            continue

        rows.append(
            {
                "frame_index": frame_idx,
                "filename": image_name,
                "track_id": target_id,
                "pixel_x": round(float(u), 3),
                "pixel_y": round(float(v), 3),
                "world_x": round(float(point_world[0]), 6),
                "world_y": round(float(point_world[1]), 6),
                "world_z": round(float(point_world[2]), 6),
                "confidence": round(float(target_det["confidence"]), 6),
            }
        )

        if frame_idx % 25 == 0:
            print(f"[{frame_idx}/{len(image_names)}] collected {len(rows)} points")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "filename",
                "track_id",
                "pixel_x",
                "pixel_y",
                "world_x",
                "world_y",
                "world_z",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved trajectory:")
    print(output_csv)
    print(f"Total points: {len(rows)}")


if __name__ == "__main__":
    main()
