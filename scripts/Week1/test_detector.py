import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.tracking.detector import YOLOTracker
from src.utils.data_loader import DataLoader


transforms_path = PROJECT_ROOT / "results/colmap_output/transforms.json"
images_dir = PROJECT_ROOT / "results/colmap_output/images"

loader = DataLoader(transforms_path)
image_names = loader.get_image_paths()

tracker = YOLOTracker(
    model_name="yolov8x.pt",
    classes=[2, 7],
    confidence=0.1,
    image_size=1280,
    tracker="botsort.yaml",
)

for image_name in image_names[:10]:
    image_path = images_dir / image_name

    detections = tracker.track_image_path(image_path)

    print(f"\nImage: {image_name}")
    print(f"Detections: {len(detections)}")

    for det in detections:
        print(det)
