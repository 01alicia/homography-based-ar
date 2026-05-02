"""Command-line interface for the computer vision pipeline."""

import argparse

from src.pipeline import process_image_set, process_video


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Camera calibration and augmented reality pipeline"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["images", "video"],
        required=True,
        help="Processing mode: 'images' or 'video'.",
    )

    parser.add_argument(
        "--pattern",
        type=int,
        default=1,
        help="Pattern identifier (default: 1).",
    )

    parser.add_argument(
        "--cube-size",
        type=int,
        default=3,
        help="Size of the projected cube (default: 3).",
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Frame step for video processing (default: 2).",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display windows.",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point of the application."""
    args = parse_arguments()

    display = not args.no_display

    if args.mode == "images":
        process_image_set(
            pattern=args.pattern,
            cube_size=args.cube_size,
            display=display,
        )

    elif args.mode == "video":
        process_video(
            pattern=args.pattern,
            cube_size=args.cube_size,
            frame_step=args.frame_step,
            display=display,
        )


if __name__ == "__main__":
    main()