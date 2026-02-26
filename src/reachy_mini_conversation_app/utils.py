import logging
import argparse
import warnings
from typing import Any, Tuple

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker


def parse_args() -> Tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker-mediapipe",
        default=False,
        help="Use mediapipe instead of yolo",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable all camera usage")
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")

    return parser.parse_known_args()


def handle_vision_stuff(args: argparse.Namespace, current_robot: ReachyMini) -> Tuple[CameraWorker | None, Any, Any]:
    """Initialize camera, head tracker, camera worker, and vision manager."""
    camera_worker = None
    head_tracker = None
    vision_manager = None

    if not args.no_camera:
        if args.head_tracker_mediapipe:
            from reachy_mini_toolbox.vision import (
                HeadTracker,
            )

            head_tracker = HeadTracker()
        else:
            from reachy_mini_conversation_app.vision.yolo_head_tracker import (
                HeadTracker,
            )

            head_tracker = HeadTracker()

        # Initialize camera worker
        camera_worker = CameraWorker(current_robot, head_tracker)

        # Initialize vision manager only if local vision is requested
        try:
            from reachy_mini_conversation_app.vision.processors import (
                initialize_vision_manager,
            )

            vision_manager = initialize_vision_manager(camera_worker)
        except ImportError as e:
            raise ImportError(
                "To use vision, please verify dependencies: pip install -r requirements.txt",
            ) from e

    return camera_worker, head_tracker, vision_manager


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger
