from typing import Optional
import numpy as np


def estimate_ground_y(
    K: np.ndarray,
    extrinsics: dict,
    cx: float,
    cy: float,
    search_min_offset: float = 0.01,
    search_max_offset: float = 1.0,
    search_step: float = 0.01,
) -> float:
    """
    Estimate ground plane Y by finding a Y-level where center rays
    from multiple cameras converge most tightly in XZ top-down space.
    """

    K_inv = np.linalg.inv(K)

    cam_positions = np.array([
        c2w[:3, 3] for c2w in extrinsics.values()
    ])

    cam_y_min = cam_positions[:, 1].min()

    best_ground_y = cam_y_min - search_min_offset
    best_spread = np.inf

    for ground_y in np.arange(
        cam_y_min - search_min_offset,
        cam_y_min - search_max_offset,
        -search_step,
    ):
        hits = []

        for c2w in list(extrinsics.values())[::5]:
            pt = pixel_to_ground_ray(
                K=K,
                c2w=c2w,
                u=cx,
                v=cy,
                ground_y=ground_y,
            )

            if pt is not None:
                hits.append(pt[[0, 2]])

        if len(hits) < 10:
            continue

        spread = np.array(hits).std(axis=0).mean()

        if spread < best_spread:
            best_spread = spread
            best_ground_y = ground_y

    return float(best_ground_y)


def pixel_to_ground_ray(
    K: np.ndarray,
    c2w: np.ndarray,
    u: float,
    v: float,
    ground_y: float,
) -> Optional[np.ndarray]:
    """
    Project a pixel into 3D by intersecting its camera ray
    with horizontal ground plane Y = ground_y.
    """

    K_inv = np.linalg.inv(K)

    origin = c2w[:3, 3]

    pixel_h = np.array([u, v, 1.0], dtype=np.float32)

    ray_camera = K_inv @ pixel_h

    ray_world = c2w[:3, :3] @ ray_camera

    ray_world = ray_world / np.linalg.norm(ray_world)

    if abs(ray_world[1]) < 1e-8:
        return None

    distance = (ground_y - origin[1]) / ray_world[1]

    if distance <= 0:
        return None

    point_world = origin + distance * ray_world

    return point_world


def bbox_bottom_center(bbox_xywh):
    """
    Convert YOLO bbox xywh to bottom-center pixel.
    xywh = [x_center, y_center, width, height]
    """

    x, y, w, h = bbox_xywh

    u = float(x)
    v = float(y + h / 2.0)

    return u, v


def bbox_center(bbox_xywh):
    """
    Convert YOLO bbox xywh to center pixel.
    """

    x, y, w, h = bbox_xywh

    return float(x), float(y)
