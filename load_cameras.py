import json
import numpy as np
import os

def load_camera_matrices(json_path):
    print(f"Loading camera data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Build the Intrinsic Matrix (K)
    # These values describe the physical drone lens
    fl_x = data['fl_x']
    fl_y = data['fl_y']
    cx = data['cx']
    cy = data['cy']

    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
    ])
    
    print("\n--- Intrinsic Matrix (K) ---")
    print(np.round(K, 2))

    # 2. Extract Extrinsic Matrices for every frame
    extrinsic_matrices = {}
    
    for frame in data['frames']:
        # The filename in the JSON looks like "images/frame_00001.png"
        # We just want the base name "frame_00001.png" to use as a dictionary key
        frame_name = os.path.basename(frame['file_path'])
        
        # The 4x4 transform_matrix from the JSON
        c2w_matrix = np.array(frame['transform_matrix'])
        
        extrinsic_matrices[frame_name] = c2w_matrix

    print(f"\nSuccessfully loaded extrinsic matrices for {len(extrinsic_matrices)} frames.")
    return K, extrinsic_matrices

# Test the function
json_file = 'results/colmap_output/transforms.json'
if os.path.exists(json_file):
    K, extrinsics = load_camera_matrices(json_file)
    
    # Let's look at the extrinsic matrix for the very first frame
    first_frame_key = list(extrinsics.keys())[0]
    print(f"\n--- Extrinsic Matrix (Camera-to-World) for {first_frame_key} ---")
    print(np.round(extrinsics[first_frame_key], 4))
else:
    print(f"Could not find {json_file}.")
