"""High-level processing pipeline for image sets and videos."""

from pathlib import Path

import cv2

from src.calibration import calibrate_from_image_set, calibrate_from_video, create_chessboard_object_points
from src.projection import process_frame
from src.metrics import plot_errors


IMAGE_PATTERNS = {
    1: {"pattern_size": (24, 17), "image_format": "png"},
    2: {"pattern_size": (31, 23), "image_format": "jpg"},
}

VIDEO_PATTERNS = {
    1: {"pattern_size": (9, 7), "video_format": "mp4"},
    2: {"pattern_size": (9, 6), "video_format": "avi"},
    3: {"pattern_size": (15, 10), "video_format": "mp4"},
}


def process_image_set(
    pattern: int = 1,
    cube_size: int = 3,
    load_existing_calibration: bool = True,
    base_dir: str | Path = "data/images",
    display: bool = True,
) -> None:
    """Process an image set and project a cube on each detected chessboard.

    Args:
        pattern: Image pattern identifier.
        cube_size: Cube height in chessboard cell units.
        load_existing_calibration: Whether to reuse saved calibration coefficients.
        base_dir: Root directory containing image pattern folders.
        display: Whether to show processed frames in an OpenCV window.

    Raises:
        ValueError: If the pattern identifier is unknown.
    """
    if pattern not in IMAGE_PATTERNS:
        raise ValueError(f"Unknown image pattern: {pattern}")

    config = IMAGE_PATTERNS[pattern]
    pattern_size = config["pattern_size"]
    image_format = config["image_format"]

    dataset_dir = Path(base_dir) / f"pattern{pattern}"

    output_dir = Path("outputs/images") / f"pattern{pattern}"
    stats_dir = output_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    _, camera_matrix, distortion, _, _, _ = calibrate_from_image_set(
        dataset_dir,
        image_format,
        pattern_size,
        load_existing_calibration,
        output_dir=output_dir,
    )

    object_points = create_chessboard_object_points(pattern_size)

    input_paths = sorted((dataset_dir / "input").glob(f"*.{image_format}"))
    output_dir = Path("outputs/images") / f"pattern{pattern}"
    stats_dir = output_dir / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    projection_errors: list[float] = []
    homography_errors: list[float] = []

    for frame_index, image_path in enumerate(input_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        processed, projection_error, homography_error = process_frame(
            image,
            pattern_size,
            camera_matrix,
            distortion,
            object_points,
            cube_size,
        )

        if projection_error is not None:
            projection_errors.append(projection_error)
        if homography_error is not None:
            homography_errors.append(homography_error)

        cv2.imwrite(str(output_dir / f"frame_{frame_index:03d}.png"), processed)

        if display:
            cv2.imshow("Projected cube", processed)
            cv2.waitKey(30)

    if display:
        cv2.destroyAllWindows()

    plot_errors(
        projection_errors,
        "Projection error",
        "Projection error over frames",
        "Frame",
        "Projection error",
        stats_dir / "projection.png",
    )
    plot_errors(
        homography_errors,
        "Homography error",
        "Homography error over frames",
        "Frame",
        "Homography error",
        stats_dir / "homography.png",
    )


def process_video(
    pattern: int = 1,
    cube_size: int = 3,
    frame_step: int = 2,
    max_calibration_frames: int = 50,
    base_dir: str | Path = "data/videos",
    display: bool = True,
) -> None:
    """Process a video and project a cube on detected chessboard frames.

    Args:
        pattern: Video pattern identifier.
        cube_size: Cube height in chessboard cell units.
        frame_step: Number of frames skipped between two processed frames.
        max_calibration_frames: Maximum number of frames used for calibration.
        base_dir: Root directory containing video pattern folders.
        display: Whether to show processed frames in an OpenCV window.

    Raises:
        ValueError: If the pattern identifier is unknown.
        FileNotFoundError: If the input video cannot be opened.
    """
    if pattern not in VIDEO_PATTERNS:
        raise ValueError(f"Unknown video pattern: {pattern}")

    config = VIDEO_PATTERNS[pattern]
    pattern_size = config["pattern_size"]
    video_format = config["video_format"]

    dataset_dir = Path(base_dir) / f"pattern{pattern}"
    input_video_path = dataset_dir / f"input.{video_format}"

    output_dir = Path("outputs/videos") / f"pattern{pattern}"
    output_video_path = output_dir / "output.mp4"
    stats_dir = output_dir / "stats"

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix, distortion, _, _ = calibrate_from_video(
        dataset_dir,
        video_format,
        pattern_size,
        frame_step,
        max_calibration_frames,
        output_dir
    )

    object_points = create_chessboard_object_points(pattern_size)

    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, fps // frame_step),
        (frame_width, frame_height),
    )

    projection_errors: list[float] = []
    homography_errors: list[float] = []

    frame_index = 0

    while capture.isOpened():
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_step == 0:
            processed, projection_error, homography_error = process_frame(
                frame,
                pattern_size,
                camera_matrix,
                distortion,
                object_points,
                cube_size,
            )

            if projection_error is not None:
                projection_errors.append(projection_error)
            if homography_error is not None:
                homography_errors.append(homography_error)

            writer.write(processed)

            if display:
                cv2.imshow("Projected cube", processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_index += 1

    capture.release()
    writer.release()

    if display:
        cv2.destroyAllWindows()

    plot_errors(
        projection_errors,
        "Projection error",
        "Projection error over frames",
        "Frame",
        "Projection error",
        stats_dir / "projection.png",
    )
    plot_errors(
        homography_errors,
        "Homography error",
        "Homography error over frames",
        "Frame",
        "Homography error",
        stats_dir / "homography.png",
    )