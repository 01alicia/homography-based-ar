"""Camera calibration utilities for chessboard-based datasets."""

from pathlib import Path

import cv2 as cv
import numpy as np


def create_chessboard_object_points(pattern_size: tuple[int, int]) -> np.ndarray:
    """Create ideal 3D object points for a planar chessboard.

    Args:
        pattern_size: Number of inner corners as `(columns, rows)`.

    Returns:
        Array of shape `(N, 3)` containing chessboard points on the Z=0 plane.
    """
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[
        0:pattern_size[0],
        0:pattern_size[1],
    ].T.reshape(-1, 2)
    return object_points


def calibrate_from_image_set(
    dataset_dir: str | Path,
    image_format: str,
    pattern_size: tuple[int, int],
    load_existing: bool = True,
    output_dir: str | Path | None = None,
) -> tuple[float, np.ndarray, np.ndarray, int, list[np.ndarray], list[np.ndarray]]:
    """Calibrate a camera from a set of chessboard images.

    Args:
        dataset_dir: Directory containing an `input/` folder.
        image_format: Image extension, for example `"png"` or `"jpg"`.
        pattern_size: Number of inner corners as `(columns, rows)`.
        load_existing: Whether to load existing calibration coefficients if available.
        output_dir: The directory where to save the outputs.

    Returns:
        Calibration RMS error, camera matrix, distortion coefficients, percentage
        of usable images, rotation vectors, and translation vectors.

    Raises:
        FileNotFoundError: If no input image is found.
        RuntimeError: If no chessboard can be detected.
    """

    output_dir = Path(output_dir) if output_dir is not None else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficients_path = output_dir / "calibration_coefficients.yml"

    if load_existing and coefficients_path.exists():
        return load_coefficients(coefficients_path)

    image_paths = sorted((dataset_dir / "input").glob(f"*.{image_format}"))
    if not image_paths:
        raise FileNotFoundError(f"No .{image_format} images found in {dataset_dir / 'input'}.")

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    reference_points = create_chessboard_object_points(pattern_size)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    frame_size: tuple[int, int] | None = None

    calibration_output_dir = output_dir / "calibration"
    calibration_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        if frame_size is None:
            frame_size = image.shape[1::-1]

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH
            + cv.CALIB_CB_NORMALIZE_IMAGE
            + cv.CALIB_CB_FAST_CHECK,
        )

        if not found:
            print(f"Chessboard not detected in {image_path.name}")
            continue

        refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(reference_points)
        image_points.append(refined_corners)

        annotated = cv.drawChessboardCorners(image.copy(), pattern_size, refined_corners, found)
        cv.imwrite(str(calibration_output_dir / f"calibration_{len(object_points):03d}.png"), annotated)

    if not object_points or frame_size is None:
        raise RuntimeError("No chessboard was detected. Calibration cannot be performed.")

    rms, camera_matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
        object_points,
        image_points,
        frame_size,
        None,
        None,
    )

    accuracy = int((len(object_points) / len(image_paths)) * 100)
    save_coefficients(rms, camera_matrix, distortion, rvecs, tvecs, accuracy, coefficients_path)

    return rms, camera_matrix, distortion, accuracy, rvecs, tvecs


def calibrate_from_video(
    video_dir: str | Path,
    video_format: str,
    pattern_size: tuple[int, int],
    frame_step: int = 10,
    max_frames: int = 100,
    output_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Calibrate a camera from chessboard detections in a video.

    Args:
        video_dir: Directory containing `input.<video_format>`.
        video_format: Video extension, for example `"mp4"` or `"avi"`.
        pattern_size: Number of inner corners as `(columns, rows)`.
        frame_step: Number of frames skipped between two processed frames.
        max_frames: Maximum number of valid chessboard frames used for calibration.
        output_dir: The directory where to save the outputs.

    Returns:
        Camera matrix, distortion coefficients, rotation vectors, and translation vectors.

    Raises:
        FileNotFoundError: If the video cannot be opened.
        RuntimeError: If no chessboard is detected.
    """
    video_dir = Path(video_dir)
    video_path = video_dir / f"input.{video_format}"

    output_dir = Path(output_dir) if output_dir is not None else video_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients_path = output_dir / "calibration_coefficients.yml"

    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    reference_points = create_chessboard_object_points(pattern_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    frame_index = 0
    processed_frames = 0
    gray = None

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_step == 0:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv.CALIB_CB_ADAPTIVE_THRESH
                + cv.CALIB_CB_NORMALIZE_IMAGE
                + cv.CALIB_CB_FAST_CHECK,
            )

            if found:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                refined_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(refined_corners)
                object_points.append(reference_points)
                processed_frames += 1

            if processed_frames >= max_frames:
                break

        frame_index += 1

    capture.release()

    if not object_points or gray is None:
        raise RuntimeError("No chessboard was detected in the video.")

    rms, camera_matrix, distortion, rvecs, tvecs = cv.calibrateCamera(
        object_points,
        image_points,
        gray.shape[::-1],
        None,
        None,
    )

    total_processed = max(1, frame_index // frame_step)
    accuracy = int((processed_frames / total_processed) * 100)
    save_coefficients(rms, camera_matrix, distortion, rvecs, tvecs, accuracy, coefficients_path)

    return camera_matrix, distortion, rvecs, tvecs


def save_coefficients(
    rms: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    accuracy: int,
    path: str | Path,
) -> None:
    """Save calibration coefficients to an OpenCV YAML file.

    Args:
        rms: Calibration RMS reprojection error.
        camera_matrix: Intrinsic camera matrix.
        distortion: Distortion coefficients.
        rvecs: Rotation vectors.
        tvecs: Translation vectors.
        accuracy: Percentage of usable calibration frames.
        path: Output YAML file path.
    """
    file_storage = cv.FileStorage(str(path), cv.FILE_STORAGE_WRITE)
    file_storage.write("RMS", float(rms))
    file_storage.write("K", camera_matrix)
    file_storage.write("D", distortion)
    file_storage.write("Accuracy", int(accuracy))
    file_storage.write("RVecs", np.array(rvecs))
    file_storage.write("TVecs", np.array(tvecs))
    file_storage.release()


def load_coefficients(
    path: str | Path,
) -> tuple[float, np.ndarray, np.ndarray, int, list[np.ndarray], list[np.ndarray]]:
    """Load calibration coefficients from an OpenCV YAML file.

    Args:
        path: YAML file containing calibration coefficients.

    Returns:
        Calibration RMS error, camera matrix, distortion coefficients, accuracy,
        rotation vectors, and translation vectors.
    """
    file_storage = cv.FileStorage(str(path), cv.FILE_STORAGE_READ)

    rms = file_storage.getNode("RMS").real()
    camera_matrix = file_storage.getNode("K").mat()
    distortion = file_storage.getNode("D").mat()
    accuracy = int(file_storage.getNode("Accuracy").real())

    rvecs_node = file_storage.getNode("RVecs")
    tvecs_node = file_storage.getNode("TVecs")

    rvecs = [np.asarray(rvec) for rvec in rvecs_node.mat()] if not rvecs_node.empty() else []
    tvecs = [np.asarray(tvec) for tvec in tvecs_node.mat()] if not tvecs_node.empty() else []

    file_storage.release()
    return rms, camera_matrix, distortion, accuracy, rvecs, tvecs