import os
os.environ['QT_LOGGING_RULES'] = "qt.qpa.fonts.warning=false"
os.environ['QT_LOGGING_RULES'] = "*=false"
os.environ['QT_DEBUG_PLUGINS'] = "0"

import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Hardware check: Running on {device.upper()}")

model = YOLO('yolov8n.pt')
#model = YOLO('yolov8x.pt')

input_video_path = 'data/demo2.mp4'
filename = os.path.basename(input_video_path)
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True) 

output_video_path = os.path.join(output_dir, filename)
blank_video_path = os.path.join(output_dir, 'blank_bf.mp4')

print(f"Reading video from: {input_video_path}")
print(f"Standard output: {output_video_path}")
print(f"Blank background output: {blank_video_path}")

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_standard = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
out_blank = cv2.VideoWriter(blank_video_path, fourcc, fps, (width, height))

# ---------------------------------------------------------
# TRAJECTORY STORAGE
# ---------------------------------------------------------
trail_length = 60
# For dynamic video trail (disappearing memory)
track_history = deque(maxlen=trail_length) 
# NEW: For final static image (infinite memory)
complete_trajectory_coords = [] 

target_id = None 
seen_ids = set()

print("Processing video...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)

    results = model.track(frame, persist=True, classes=[2, 7], tracker="botsort.yaml", verbose=False, device=device, conf=0.1, imgsz=1280)
    
    # ---------------------------------------------------------
    # INTERACTIVE SELECTION (GUI-Based)
    # ---------------------------------------------------------
    if target_id is None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        current_ids_set = set(track_ids)
        new_ids = current_ids_set - seen_ids
        
        if new_ids:
            display_frame = frame.copy()
            
            for box, t_id in zip(boxes, track_ids):
                x_center, y_center, w, h = box
                top_left = (int(x_center - w/2), int(y_center - h/2))
                bottom_right = (int(x_center + w/2), int(y_center + h/2))
                
                cv2.rectangle(display_frame, top_left, bottom_right, (255, 0, 0), 2)
                cv2.putText(display_frame, f"ID: {t_id}", (top_left[0], top_left[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            print(f"\nPaused. New IDs detected: {list(new_ids)}")
            print("Click on the video window. Press 's' to skip, or type the ID and press Enter.")
            
            user_input_str = ""
            while True:
                prompt_frame = display_frame.copy()
                cv2.putText(prompt_frame, "PAUSED - Press 's' to skip forward", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(prompt_frame, f"Type ID and press ENTER: {user_input_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Select Your Car", prompt_frame)
                
                key = cv2.waitKey(33) & 0xFF
                
                if key == ord('s'):
                    seen_ids.update(current_ids_set)
                    print("Skipping to next new ID...")
                    break
                elif ord('0') <= key <= ord('9'):
                    user_input_str += chr(key)
                elif key == 8 or key == 127: # Backspace
                    user_input_str = user_input_str[:-1]
                elif key == 13 or key == 10: # Enter
                    if user_input_str.isdigit():
                        selected = int(user_input_str)
                        if selected in track_ids:
                            target_id = selected
                            print(f"Locked onto ID: {target_id}")
                            cv2.destroyWindow("Select Your Car")
                            break
                        else:
                            print(f"ID {selected} is not valid. Try again.")
                            user_input_str = ""

    # ---------------------------------------------------------
    # TRACKING & DRAWING PHASE (Videos)
    # ---------------------------------------------------------
    if target_id is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            if track_id == target_id:
                x_center, y_center, w, h = box
                center_point = (int(x_center), int(y_center))
                
                # Add to both storage locations
                track_history.append(center_point) # temporary
                complete_trajectory_coords.append(center_point) # NEW: permanent
                
                top_left = (int(x_center - w/2), int(y_center - h/2))
                bottom_right = (int(x_center + w/2), int(y_center + h/2))
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # Drawing dynamic disappearing trail in videos
        for i in range(1, len(track_history)):
            ratio = i / len(track_history)
            blue_green_val = int(200 * (1 - ratio)) 
            color = (blue_green_val, blue_green_val, 255) 
            thickness = int(4 * ratio) + 1
            
            cv2.line(frame, track_history[i - 1], track_history[i], color, thickness=thickness)
            cv2.line(blank_frame, track_history[i - 1], track_history[i], color, thickness=thickness)

    out_standard.write(frame)
    out_blank.write(blank_frame)

# Clean up video writers
cap.release()
out_standard.release()
out_blank.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# NEW: GENERATE AND SAVE COMPLETE TRAJECTORY IMAGE
# ---------------------------------------------------------
if target_id is not None and complete_trajectory_coords:
    print(f"Video loop finished. Generating complete trajectory image...")
    
    # Define output image path
    trajectory_img_path = os.path.join(output_dir, 'complete_trajectory.png')
    
    # Create black canvas matching video dimensions
    final_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw entire trajectory with a static gradient (Red to pink/white)
    # based on total length to show path progression over time.
    total_points = len(complete_trajectory_coords)
    for i in range(1, total_points):
        # Ratio based on entire history, not just 60 points
        ratio = i / total_points
        
        # Start of video = pinkish, end of video = solid red
        blue_green_val = int(200 * (1 - ratio)) 
        color = (blue_green_val, blue_green_val, 255) 
        
        # Thin solid line for the final image to avoid clutter
        cv2.line(final_img, complete_trajectory_coords[i - 1], complete_trajectory_coords[i], color, thickness=2)

    # Add text to confirm which ID this path belongs to
    cv2.putText(final_img, f"Full Trajectory for Car ID: {target_id}", (20, height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Save the static image
    cv2.imwrite(trajectory_img_path, final_img)
    print(f"Complete trajectory image saved as: {trajectory_img_path}")
else:
    print("No target selected, skipping trajectory image generation.")

print(f"All processing complete! Check the '{output_dir}' folder.")
