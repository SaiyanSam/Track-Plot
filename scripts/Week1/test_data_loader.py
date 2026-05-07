import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loader import DataLoader

loader = DataLoader(
    "results/colmap_output/transforms.json"
)

K = loader.get_intrinsics()

extrinsics = loader.get_extrinsics()

images = loader.get_image_paths()

print("\n========== INTRINSICS ==========")
print(K)

print(f"\nLoaded {len(images)} images")

first_key = list(extrinsics.keys())[0]

print(f"\nFirst image: {first_key}")

print("\n========== FIRST EXTRINSIC ==========")
print(extrinsics[first_key])
