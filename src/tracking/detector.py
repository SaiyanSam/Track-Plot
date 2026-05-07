from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import torch
from ultralytics import YOLO


class YOLOTracker:
    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        classes: Optional[List[int]] = None,
        confidence: float = 0.1,
        image_size: int = 1280,
        tracker: str = "botsort.yaml",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.classes = classes if classes is not None else [2, 7]  # car, truck
        self.confidence = confidence
        self.image_size = image_size
        self.tracker = tracker
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[YOLOTracker] Loading {model_name} on {self.device}")
        self.model = YOLO(model_name)

    def track_frame(self, frame) -> List[Dict[str, Any]]:
        """
        Runs YOLO tracking on one image/frame.

        Returns a list of detections:
        [
            {
                "track_id": int,
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int
            }
        ]
        """

        results = self.model.track(
            frame,
            persist=True,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
            device=self.device,
            conf=self.confidence,
            imgsz=self.image_size,
        )

        if len(results) == 0:
            return []

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            return []

        boxes_xywh = result.boxes.xywh.cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()

        detections = []

        for xywh, xyxy, tid, cid, conf in zip(
            boxes_xywh, boxes_xyxy, track_ids, class_ids, confs
        ):
            detections.append(
                {
                    "track_id": int(tid),
                    "bbox_xywh": [float(v) for v in xywh],
                    "bbox_xyxy": [float(v) for v in xyxy],
                    "confidence": float(conf),
                    "class_id": int(cid),
                }
            )

        return detections

    def track_image_path(self, image_path: str) -> List[Dict[str, Any]]:
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.track_frame(frame)
