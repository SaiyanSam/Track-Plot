import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse

# 1. Setup Command Line Arguments
parser = argparse.ArgumentParser(description="Real-Time Drone Telemetry Visualizer")
parser.add_argument('--duration', type=float, default=26.0, 
                    help='Duration of the original video in seconds (default: 28.0)')
args = parser.parse_args()

# 2. Load the JSON Data
json_path = 'results/colmap_output/transforms.json'
print(f"Loading flight data from: {json_path}")

if not os.path.exists(json_path):
    print("Error: Could not find transforms.json!")
    exit()

with open(json_path, 'r') as f:
    data = json.load(f)

# Sort frames chronologically
frames = sorted(data['frames'], key=lambda x: x['file_path'])

# 3. Storage for coordinates and directions
x_coords, y_coords = [], []
dx_coords, dy_coords = [], [] 

for frame in frames:
    c2w_matrix = np.array(frame['transform_matrix'])
    
    # Extract Translation (Drone's position)
    pos_x = c2w_matrix[0, 3]
    pos_y = c2w_matrix[1, 3]
    x_coords.append(pos_x)
    y_coords.append(pos_y)
    
    # Extract Camera Forward Vector
    forward_vector = -c2w_matrix[:3, 2] 
    
    dx = forward_vector[0]
    dy = forward_vector[1]
    
    length = np.hypot(dx, dy)
    if length > 1e-6:
        dx /= length
        dy /= length
        
    dx_coords.append(dx)
    dy_coords.append(dy)

# 4. Calculate the perfect playback speed
total_frames = len(x_coords)
# Total time in ms divided by number of frames
interval_ms = int((args.duration * 1000) / total_frames) 

print(f"\n--- Synchronization Stats ---")
print(f"Total Extracted Frames: {total_frames}")
print(f"Target Video Duration:  {args.duration} seconds")
print(f"Calculated Interval:    {interval_ms} ms per frame\n")

# 5. Set up the Radar Screen
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"Real-Time Top-Down Drone Position & Orientation", fontsize=14, fontweight='bold')
ax.set_xlabel("World X")
ax.set_ylabel("World Y")

buffer_x = (max(x_coords) - min(x_coords)) * 0.1
buffer_y = (max(y_coords) - min(y_coords)) * 0.1
ax.set_xlim(min(x_coords) - buffer_x, max(x_coords) + buffer_x)
ax.set_ylim(min(y_coords) - buffer_y, max(y_coords) + buffer_y)
ax.grid(True, linestyle='--', alpha=0.6)

# 6. Initialize graphic elements
trail, = ax.plot([], [], 'b-', alpha=0.4, linewidth=2, label='Flight Path')
drone_dot, = ax.plot([], [], 'ro', markersize=8, label='Drone')
arrow = ax.quiver(x_coords[0], y_coords[0], dx_coords[0], dy_coords[0], 
                  color='red', pivot='tail', scale=12, width=0.006)
#ax.legend()

# 7. The Real-Time Animation Loop
def update(frame_idx):
    trail.set_data(x_coords[:frame_idx+1], y_coords[:frame_idx+1])
    drone_dot.set_data([x_coords[frame_idx]], [y_coords[frame_idx]])
    
    arrow.set_offsets(np.column_stack([x_coords[frame_idx], y_coords[frame_idx]]))
    arrow.set_UVC(dx_coords[frame_idx], dy_coords[frame_idx])
    
    return trail, drone_dot, arrow

print("Launching synced playback. Hit play on your video!")
ani = animation.FuncAnimation(
    fig, update, frames=total_frames, interval=interval_ms, blit=False
)

plt.show()
