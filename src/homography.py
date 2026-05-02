"""Homography estimation and pose recovery utilities."""

import cv2
import numpy as np
from scipy.linalg import svd


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize 2D points using mean-centering and standard deviation scaling.

    Args:
        points: Array of shape `(N, 2)`.

    Returns:
        Normalized points and the associated 3x3 normalization matrix.

    Raises:
        ValueError: If the point distribution is degenerate.
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")

    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)

    if np.any(std == 0):
        raise ValueError("Degenerate points: standard deviation is zero.")

    transform = np.array(
        [
            [1 / std[0], 0, -mean[0] / std[0]],
            [0, 1 / std[1], -mean[1] / std[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    homogeneous_points = np.column_stack((points, np.ones(len(points))))
    normalized_points = (transform @ homogeneous_points.T).T

    return normalized_points[:, :2], transform


def find_homography(image_points: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    """Estimate a planar homography from object-plane points to image points.

    Args:
        image_points: Detected 2D image points of shape `(N, 2)`.
        object_points: Planar object points of shape `(N, 2)` or `(N, 3)`.
            If 3D points are provided, only X and Y coordinates are used.

    Returns:
        A 3x3 homography matrix mapping object-plane points to image points.

    Raises:
        ValueError: If fewer than four correspondences are provided or if shapes
            are inconsistent.
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    object_points = np.asarray(object_points, dtype=np.float64)

    if object_points.ndim != 2 or object_points.shape[1] not in (2, 3):
        raise ValueError("object_points must have shape (N, 2) or (N, 3).")

    if object_points.shape[1] == 3:
        object_points = object_points[:, :2]

    if len(image_points) != len(object_points):
        raise ValueError("image_points and object_points must contain the same number of points.")

    if len(image_points) < 4:
        raise ValueError("At least four correspondences are required.")

    normalized_image_points, image_transform = normalize_points(image_points)
    normalized_object_points, object_transform = normalize_points(object_points)

    system_rows = []
    for (x_world, y_world), (x_image, y_image) in zip(normalized_object_points, normalized_image_points):
        system_rows.append([x_world, y_world, 1, 0, 0, 0, -x_image * x_world, -x_image * y_world, -x_image])
        system_rows.append([0, 0, 0, x_world, y_world, 1, -y_image * x_world, -y_image * y_world, -y_image])

    matrix_a = np.asarray(system_rows, dtype=np.float64)
    _, _, vt = svd(matrix_a)

    homography = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(image_transform) @ homography @ object_transform
    homography /= homography[2, 2]

    return homography


def decompose_homography(homography: np.ndarray, camera_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover camera pose from a planar homography.

    Args:
        homography: 3x3 homography mapping object-plane points to image points.
        camera_matrix: 3x3 intrinsic camera matrix.

    Returns:
        Rotation matrix and translation vector.

    Raises:
        ValueError: If input matrices do not have shape `(3, 3)`.
    """
    homography = np.asarray(homography, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    if homography.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3).")

    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3).")

    inv_camera = np.linalg.inv(camera_matrix)

    h1 = homography[:, 0]
    h2 = homography[:, 1]
    h3 = homography[:, 2]

    scale = 1.0 / np.linalg.norm(inv_camera @ h1)

    r1 = scale * (inv_camera @ h1)
    r2 = scale * (inv_camera @ h2)
    r3 = np.cross(r1, r2)
    translation = scale * (inv_camera @ h3)

    rotation = np.column_stack((r1, r2, r3))

    # Re-orthogonalize the rotation matrix for numerical stability.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt

    return rotation, translation


def opencv_homography(image_points: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    """Estimate a reference homography using OpenCV.

    Args:
        image_points: Detected 2D image points of shape `(N, 2)`.
        object_points: Planar object points of shape `(N, 2)` or `(N, 3)`.

    Returns:
        A 3x3 homography matrix mapping object-plane points to image points.
    """
    object_points = np.asarray(object_points, dtype=np.float64)
    if object_points.shape[1] == 3:
        object_points = object_points[:, :2]

    homography, _ = cv2.findHomography(object_points, image_points)
    return homography