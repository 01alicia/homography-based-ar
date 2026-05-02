"""3D projection and augmented reality rendering utilities."""

import cv2
import numpy as np

from src.homography import decompose_homography, find_homography, opencv_homography
from src.metrics import compute_homography_error, compute_projection_error


def create_centered_cube(pattern_size: tuple[int, int], cube_size: int = 3) -> np.ndarray:
    """Create a cube centered on the chessboard.

    Args:
        pattern_size: Number of inner corners as `(columns, rows)`.
        cube_size: Cube height in chessboard cell units.

    Returns:
        Array of shape `(8, 3)` containing cube vertices.
    """
    center_x = pattern_size[0] // 2
    center_y = pattern_size[1] // 2

    x_start = center_x - 1
    y_start = center_y - 1

    return np.array(
        [
            [x_start, y_start, 0],
            [x_start + 3, y_start, 0],
            [x_start + 3, y_start + 3, 0],
            [x_start, y_start + 3, 0],
            [x_start, y_start, -cube_size],
            [x_start + 3, y_start, -cube_size],
            [x_start + 3, y_start + 3, -cube_size],
            [x_start, y_start + 3, -cube_size],
        ],
        dtype=np.float64,
    )


def project_points(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project 3D points into the image plane.

    Args:
        points_3d: 3D points of shape `(N, 3)`.
        camera_matrix: 3x3 intrinsic camera matrix.
        rotation: 3x3 rotation matrix.
        translation: Translation vector of shape `(3,)`.

    Returns:
        Projected 2D points of shape `(N, 2)`.
    """
    projection_matrix = camera_matrix @ np.hstack((rotation, translation.reshape(-1, 1)))

    projected_points = []
    for point in points_3d:
        homogeneous_point = np.append(point, 1.0)
        projected = projection_matrix @ homogeneous_point
        projected /= projected[2]
        projected_points.append(projected[:2])

    return np.asarray(projected_points)


def draw_cube(image: np.ndarray, base_points: np.ndarray, top_points: np.ndarray) -> np.ndarray:
    """Draw a projected cube on an image.

    Args:
        image: Image on which the cube is drawn.
        base_points: Projected points of the cube base.
        top_points: Projected points of the cube top face.

    Returns:
        Image with the cube overlay.
    """
    output = image.copy()

    for index in range(4):
        base_start = tuple(base_points[index].astype(int))
        base_end = tuple(base_points[(index + 1) % 4].astype(int))

        top_start = tuple(top_points[index].astype(int))
        top_end = tuple(top_points[(index + 1) % 4].astype(int))

        cv2.line(output, base_start, base_end, (0, 255, 0), 2)
        cv2.line(output, top_start, top_end, (255, 0, 0), 2)
        cv2.line(output, base_start, top_start, (0, 0, 255), 2)

    return output


def process_frame(
    image: np.ndarray,
    pattern_size: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    object_points: np.ndarray,
    cube_size: int = 3,
) -> tuple[np.ndarray, float | None, float | None]:
    """Detect a chessboard and project a 3D cube onto the frame.

    Args:
        image: Input image.
        pattern_size: Number of inner corners as `(columns, rows)`.
        camera_matrix: 3x3 intrinsic camera matrix.
        distortion: Distortion coefficients.
        object_points: Chessboard object points of shape `(N, 3)`.
        cube_size: Cube height in chessboard cell units.

    Returns:
        Processed image, projection error, and homography error. Errors are
        `None` when the chessboard is not detected.
    """
    undistorted = cv2.undistort(image, camera_matrix, distortion)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )

    if not found:
        return undistorted, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    image_points = refined_corners.reshape(-1, 2)

    manual_homography = find_homography(image_points, object_points)
    reference_homography = opencv_homography(image_points, object_points)

    rotation, translation = decompose_homography(manual_homography, camera_matrix)

    cube = create_centered_cube(pattern_size, cube_size)
    base_projected = project_points(cube[:4], camera_matrix, rotation, translation)
    top_projected = project_points(cube[4:], camera_matrix, rotation, translation)

    output = draw_cube(undistorted, base_projected, top_projected)

    projection_error = compute_projection_error(
        object_points,
        manual_homography,
        camera_matrix,
        distortion,
        rotation,
        translation,
    )
    homography_error = compute_homography_error(manual_homography, reference_homography)

    return output, projection_error, homography_error