import sys
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


transforms_path = PROJECT_ROOT / "results/colmap_output/transforms.json"
images_dir = PROJECT_ROOT / "results/colmap_output/images"

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

print(f"\nEstimated ground Y: {ground_y:.6f}")

tracker = YOLOTracker(
    model_name="yolov8x.pt",
    classes=[2, 7],
    confidence=0.1,
    image_size=1280,
    tracker="botsort.yaml",
)

for image_name in image_names[:10]:
    image_path = images_dir / image_name

    if image_name not in extrinsics:
        continue

    detections = tracker.track_image_path(image_path)

    print(f"\nImage: {image_name}")
    print(f"Detections: {len(detections)}")

    for det in detections:
        u, v = bbox_bottom_center(det["bbox_xywh"])

        pt = pixel_to_ground_ray(
            K=K,
            c2w=extrinsics[image_name],
            u=u,
            v=v,
            ground_y=ground_y,
        )

        if pt is not None:
            print(
                f"ID={det['track_id']} "
                f"pixel=({u:.1f}, {v:.1f}) "
                f"world=({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
            )
        else:
            print(f"ID={det['track_id']} projection failed")
