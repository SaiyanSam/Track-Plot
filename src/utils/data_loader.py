import json
from pathlib import Path
import numpy as np


class DataLoader:

    def __init__(self, transforms_json_path):

        self.transforms_json_path = Path(transforms_json_path)

        if not self.transforms_json_path.exists():
            raise FileNotFoundError(
                f"Could not find: {self.transforms_json_path}"
            )

        with open(self.transforms_json_path, "r") as f:
            self.data = json.load(f)

        self.frames = sorted(
            self.data["frames"],
            key=lambda x: x["file_path"]
        )

    def get_intrinsics(self):

        K = np.array([
            [self.data["fl_x"], 0, self.data["cx"]],
            [0, self.data["fl_y"], self.data["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def get_extrinsics(self):

        extrinsics = {}

        for frame in self.frames:

            filename = Path(frame["file_path"]).name

            extrinsics[filename] = np.array(
                frame["transform_matrix"],
                dtype=np.float32
            )

        return extrinsics

    def get_image_paths(self):

        return [
            Path(frame["file_path"]).name
            for frame in self.frames
        ]

    def get_frame_count(self):

        return len(self.frames)
