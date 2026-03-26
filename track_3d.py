import os
import json
import cv2
import csv
import numpy as np
from ultralytics import YOLO
import torch

os.environ['QT_LOGGING_RULES'] = "*=false"

def pixel_to_world(u, v, K, c2w_matrix, ground_z=0.0):
    R_c2w = c2w_matrix[:3, :3]
    camera_origin = c2w_matrix[:3, 3] 
    
    pixel_homogeneous = np.array([u, v, 1.0])
    K_inv = np.linalg.inv(K)
    ray_dir_camera = K_inv @ pixel_homogeneous
    
    ray_dir_world = R_c2w @ ray_dir_camera
    ray_dir_world = ray_dir_world / np.linalg.norm(ray_dir_world)
    
    if abs(ray_dir_world[2]) < 1e-6:
        return None
        
    distance_to_ground = (ground_z - camera_origin[2]) / ray_dir_world[2]
    
    # Filter out backward rays or impossibly distant intersections
    if distance_to_ground < 0 or distance_to_ground > 100.0:
        return None
        
    return camera_origin + (distance_to_ground * ray_dir_world)

def smooth_trajectory(points_3d, window_size=11):
    if len(points_3d) < window_size:
        return points_3d
    
    # Stage 1: Median filter to remove teleportation spikes
    despiked = []
    for i in range(len(points_3d)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(points_3d), i + window_size // 2 + 1)
        window = points_3d[start_idx:end_idx]
        
        med_x = np.median([p[0] for p in window])
        med_y = np.median([p[1] for p in window])
        med_z = np.median([p[2] for p in window])
        despiked.append(np.array([med_x, med_y, med_z]))
        
    # Stage 2: Moving average to smooth the valid path
    smoothed = []
    for i in range(len(despiked)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(despiked), i + window_size // 2 + 1)
        window = despiked[start_idx:end_idx]
        
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        avg_z = sum(p[2] for p in window) / len(window)
        smoothed.append(np.array([avg_x, avg_y, avg_z]))
        
    return smoothed

json_path = 'results/colmap_output/transforms.json'
print(f"Loading matrices from {json_path}")

with open(json_path, 'r') as f:
    data = json.load(f)

K = np.array([
    [data['fl_x'], 0, data['cx']],
    [0, data['fl_y'], data['cy']],
    [0, 0, 1]
])

extrinsics = {}
for frame in data['frames']:
    filename = os.path.basename(frame['file_path'])
    extrinsics[filename] = np.array(frame['transform_matrix'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8x.pt')

image_dir = 'results/colmap_output/images'
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

target_id = None 
seen_ids = set()
raw_tracking_data = []

print(f"Processing {len(image_files)} frames...")

for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    frame = cv2.imread(img_path)
    
    if img_name not in extrinsics:
        continue
        
    c2w = extrinsics[img_name]
    results = model.track(frame, persist=True, classes=[2, 7], tracker="botsort.yaml", verbose=False, device=device, conf=0.1, imgsz=1280)
    
    # GUI for interactive ID selection
    if target_id is None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        current_ids_set = set(track_ids)
        new_ids = current_ids_set - seen_ids
        
        if new_ids:
            display_frame = frame.copy()
            for box, t_id in zip(boxes, track_ids):
                x_c, y_c, w, h = box
                top_left = (int(x_c - w/2), int(y_c - h/2))
                bottom_right = (int(x_c + w/2), int(y_c + h/2))
                cv2.rectangle(display_frame, top_left, bottom_right, (255, 0, 0), 2)
                cv2.putText(display_frame, f"ID: {t_id}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            print(f"Paused. New IDs detected: {list(new_ids)}")
            user_input_str = ""
            while True:
                prompt_frame = display_frame.copy()
                cv2.putText(prompt_frame, "Press 's' to skip", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(prompt_frame, f"Type ID & ENTER: {user_input_str}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Select Target", prompt_frame)
                key = cv2.waitKey(33) & 0xFF
                
                if key == ord('s'):
                    seen_ids.update(current_ids_set)
                    break
                elif ord('0') <= key <= ord('9'):
                    user_input_str += chr(key)
                elif key == 8 or key == 127: 
                    user_input_str = user_input_str[:-1]
                elif key == 13 or key == 10: 
                    if user_input_str.isdigit() and int(user_input_str) in track_ids:
                        target_id = int(user_input_str)
                        print(f"Tracking ID: {target_id}")
                        cv2.destroyWindow("Select Target")
                        break
                    else:
                        user_input_str = ""

    # Triangulation phase
    if target_id is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            if track_id == target_id:
                x_center, y_center, w, h = box
                
                # Shift projection target to the bottom of the bounding box (tires)
                y_bottom = y_center + (h / 2.0)
                
                point_3d = pixel_to_world(float(x_center), float(y_bottom), K, c2w)
                if point_3d is not None:
                    raw_tracking_data.append((img_name, float(x_center), float(y_bottom), point_3d))

# Post-processing and export
if len(raw_tracking_data) > 0:
    print(f"Applying smoothing filter to {len(raw_tracking_data)} points...")
    
    raw_points_3d = [data[3] for data in raw_tracking_data]
    smoothed_points_3d = smooth_trajectory(raw_points_3d, window_size=11)
    
    csv_path = 'results/car_3d_trajectory.csv'
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Filename', 'Pixel_X', 'Pixel_Y', 'World_X', 'World_Y', 'World_Z'])
        
        for data, pt in zip(raw_tracking_data, smoothed_points_3d):
            csv_writer.writerow([
                data[0], round(data[1], 1), round(data[2], 1), 
                round(pt[0], 4), round(pt[1], 4), round(pt[2], 4)
            ])
    print(f"Saved smoothed trajectory to {csv_path}")

    # Generate visual outputs
    original_video_path = 'data/demo1.mp4' 
    orig_duration = 28.0 
    if os.path.exists(original_video_path):
        cap = cv2.VideoCapture(original_video_path)
        if cap.get(cv2.CAP_PROP_FPS) > 0:
            orig_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
    radar_fps = len(smoothed_points_3d) / orig_duration

    canvas_w, canvas_h = 1920, 1080
    margin = 150 
    
    world_x = [p[0] for p in smoothed_points_3d]
    world_y = [p[1] for p in smoothed_points_3d]
    
    min_x, max_x = min(world_x), max(world_x)
    min_y, max_y = min(world_y), max(world_y)
    
    range_x = max_x - min_x or 1
    range_y = max_y - min_y or 1
    
    scale_x = (canvas_w - 2 * margin) / range_x
    scale_y = (canvas_h - 2 * margin) / range_y
    scale = min(scale_x, scale_y) 
    
    offset_x = (canvas_w - (range_x * scale)) / 2
    offset_y = (canvas_h - (range_y * scale)) / 2
    
    drawn_points = []
    for x, y in zip(world_x, world_y):
        px = int((x - min_x) * scale + offset_x)
        py = int(canvas_h - ((y - min_y) * scale + offset_y)) 
        drawn_points.append((px, py))

    video_path = 'results/car_3d_trajectory_animated.mp4'
    map_path = 'results/car_3d_trajectory_map.png'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, radar_fps, (canvas_w, canvas_h))

    total_points = len(drawn_points)
    
    for current_frame_idx in range(1, total_points):
        frame_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for i in range(1, current_frame_idx + 1):
            ratio = i / total_points
            blue_green_val = int(200 * (1 - ratio)) 
            color = (blue_green_val, blue_green_val, 255) 
            cv2.line(frame_canvas, drawn_points[i - 1], drawn_points[i], color, thickness=4)
            
        current_pos = drawn_points[current_frame_idx]
        cv2.circle(frame_canvas, current_pos, 10, (0, 0, 255), -1) 
        cv2.circle(frame_canvas, current_pos, 18, (100, 100, 255), 2) 

        cv2.putText(frame_canvas, f"Radar View | ID: {target_id}", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        progress = (current_frame_idx / total_points) * 100
        cv2.putText(frame_canvas, f"Progress: {progress:.1f}%", (40, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out_video.write(frame_canvas)
        
        if current_frame_idx == total_points - 1:
            cv2.imwrite(map_path, frame_canvas)

    out_video.release()
    print(f"Animated radar video saved to {video_path}")
    print(f"Static map saved to {map_path}")

else:
    print("Not enough data to generate outputs.")
