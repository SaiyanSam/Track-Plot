import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Hardware check: Running on {device.upper()}")

model = YOLO('yolov8x.pt')

input_video_path = 'data/demo2.mp4'
filename = os.path.basename(input_video_path)
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True) 

output_video_path = os.path.join(output_dir, filename)
blank_video_path = os.path.join(output_dir, 'blank_bf.mp4')

print(f"Reading video from: {input_video_path}")
print(f"Standard output: {output_video_path}")
print(f"Blank background output: {blank_video_path}")

# Video capture
cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_standard = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
out_blank = cv2.VideoWriter(blank_video_path, fourcc, fps, (width, height))

# The Disappearing Trail
trail_length = 60
track_history = deque(maxlen=trail_length)
target_id = None 

print("Processing video... This may take a minute.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # np.zeros fills the array with 0s (which is black in BGR color space)
    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # YOLOv8 tracking with BoT-SORT
    # UPGRADES: 
    # 1. imgsz=1280 tells YOLO to process at double the default resolution
    # 2. classes=[2, 7] tells it to look for Cars AND Trucks
    results = model.track(
        frame, 
        persist=True, 
        classes=[2, 7], 
        tracker="botsort.yaml", 
        verbose=False, 
        device=device, 
        conf=0.1,
        imgsz=1280 
    )
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        if target_id is None:
            target_id = track_ids[0]

        for box, track_id in zip(boxes, track_ids):
            if track_id == target_id:
                x_center, y_center, w, h = box
                center_point = (int(x_center), int(y_center))
                
                track_history.append(center_point)
                
                top_left = (int(x_center - w/2), int(y_center - h/2))
                bottom_right = (int(x_center + w/2), int(y_center + h/2))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Drawing the Red Trail on both frames
    for i in range(1, len(track_history)):
        ratio = i / len(track_history)
        
        # BGR format gradient: light red to solid red
        blue_green_val = int(200 * (1 - ratio)) 
        color = (blue_green_val, blue_green_val, 255) 
        
        thickness = int(4 * ratio) + 1
        
        cv2.line(frame, track_history[i - 1], track_history[i], color, thickness=thickness)
        cv2.line(blank_frame, track_history[i - 1], track_history[i], color, thickness=thickness)

    out_standard.write(frame)
    out_blank.write(blank_frame)

cap.release()
out_standard.release()
out_blank.release()
cv2.destroyAllWindows()

print(f"Tracking complete! Check the '{output_dir}' folder for both videos.")
