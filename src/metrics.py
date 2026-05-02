"""Evaluation metrics and plotting utilities."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_errors(
    data: list[float],
    label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: str | Path,
    threshold: float = 5.0,
) -> None:
    """Plot an error curve and save it as an image.

    Args:
        data: Error values to plot.
        label: Curve label.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        output_path: Path where the plot is saved.
        threshold: Optional critical threshold displayed as a dashed line.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = np.arange(len(data))

    plt.figure()
    plt.scatter(frames, data, label=label)
    plt.axhline(y=threshold, color="red", linestyle="--", label="Critical threshold")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def compute_homography_error(manual_homography: np.ndarray, reference_homography: np.ndarray) -> float | None:
    """Compute the relative RMSE between two homography matrices.

    Args:
        manual_homography: Homography estimated by the custom implementation.
        reference_homography: Homography estimated by OpenCV.

    Returns:
        Relative RMSE normalized by the reference homography norm, or `None`
        if one matrix is missing.
    """
    if manual_homography is None or reference_homography is None:
        return None

    rmse = np.sqrt(np.mean((manual_homography - reference_homography) ** 2))
    return float(rmse / np.linalg.norm(reference_homography))


def compute_projection_error(
    object_points: np.ndarray,
    homography: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> float:
    """Compare homography-based projection with OpenCV projection.

    Args:
        object_points: 3D chessboard object points of shape `(N, 3)`.
        homography: Homography mapping object-plane points to image points.
        camera_matrix: 3x3 intrinsic camera matrix.
        distortion: Distortion coefficients.
        rotation: 3x3 rotation matrix.
        translation: Translation vector.

    Returns:
        Mean Euclidean projection error in pixels.

    Raises:
        ValueError: If object points do not have shape `(N, 3)`.
    """
    object_points = np.asarray(object_points, dtype=np.float32)

    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points must have shape (N, 3).")

    planar_points_h = np.hstack([object_points[:, :2], np.ones((len(object_points), 1))])
    projected_h = (homography @ planar_points_h.T).T
    projected_h /= projected_h[:, 2:3]
    projected_h = projected_h[:, :2]

    rvec, _ = cv2.Rodrigues(rotation)
    projected_cv, _ = cv2.projectPoints(
        object_points,
        rvec,
        translation.reshape(3, 1),
        camera_matrix,
        distortion,
    )
    projected_cv = projected_cv.reshape(-1, 2)

    error = np.linalg.norm(projected_cv - projected_h, axis=1)
    return float(np.mean(error))