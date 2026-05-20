import logging
import cv2
import shutil
import os
import tempfile
import subprocess
import tomllib
from pathlib import Path
from typing import TypeAlias


RGBColor: TypeAlias = tuple[int, int, int]
ColorFilter: TypeAlias = tuple[RGBColor, RGBColor]

class ProcessedVideo:
    """Wrapper for a processed video saved in a temporary file.
    
    Provides utility methods to save the final video, merge the original audio, 
    and safely clean up the temporary files from the disk using a context manager.
    """

    def __init__(self, path: str, log_level: int) -> None:
        """Initializes the ProcessedVideo wrapper.

        Args:
            path (str): The absolute path to the temporary processed video file.
            log_level (int): The logging level inherited from the main orchestrator.
        """

        self.logger = logging.getLogger(f"{__name__}.ProcessedVideo")
        self.logger.setLevel(level=log_level)
        
        self.path = path
        self._cap = None

    def open(self) -> cv2.VideoCapture:
        """Opens the processed video file for reading.

        Returns:
            cv2.VideoCapture: An OpenCV VideoCapture object pointing to the processed video.
        """

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
        return self._cap

    def save(self, path: str) -> None:
        """Saves the processed video to the specified destination path without audio.

        Args:
            path (str): The destination file path (e.g., 'output.mp4').
        """

        self.logger.debug(f"Saving video to {path} ...")
        shutil.copy(self.path, path)
        os.chmod(path, 0o644)

    def save_with_audio(self, original_video_path: str, path: str) -> None:
        """Merges the audio track from the original video into the processed video.

        Requires FFmpeg to be installed and available in the system's PATH.

        Args:
            original_video_path (str): The path to the source video containing the audio track.
            path (str): The destination file path for the final merged video.

        Raises:
            RuntimeError: If the FFmpeg executable is not found in the system PATH.
            ValueError: If the original_video_path is empty.
            FileNotFoundError: If the original_video_path does not exist on the disk.
        """

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg executable not found in PATH. "
                "Please install it from https://ffmpeg.org/download.html "
                "and ensure it is available in your system PATH."
            )
        if not original_video_path:
            raise ValueError("original_video_path is required to merge audio")
        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"original_video_path={original_video_path} does not exist")

        self.logger.debug(f"Saving video with audio to {path}...")
        self.logger.debug(f"Merging audio from {original_video_path} into {self.path}")

        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", self.path,             # processed video (video stream)
            "-i", original_video_path,   # original video (audio stream)
            "-c:v", "copy",              # do not re-encode video
            "-c:a", "aac",               # encode audio to AAC
            "-map", "0:v:0",             # take video from first input
            "-map", "1:a:0?",            # take audio from second input (if it exists)
            temp_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            shutil.copy2(temp_path, path)
            os.chmod(path, 0o644)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        self.logger.debug(f"Audio merged successfully into {path}")

    def release(self) -> None:
        """Closes the video capture and deletes the temporary video file from the disk."""

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self.path and os.path.exists(self.path):
            self.logger.debug(f"Removing temporary file {self.path} ...")
            os.remove(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

class ProcessedGIF:
    """Wrapper for a processed animated GIF saved in a temporary file."""
    
    def __init__(self, path: str, log_level: int) -> None:
        self.logger = logging.getLogger(f"{__name__}.ProcessedGIF")
        self.logger.setLevel(level=log_level)
        self.path = path

    def save(self, dest_path: str) -> None:
        """Saves the processed GIF to the specified destination path."""
        self.logger.debug(f"Saving GIF to {dest_path} ...")
        shutil.copy(self.path, dest_path)
        os.chmod(dest_path, 0o644)

    def release(self) -> None:
        """Deletes the temporary GIF file from the disk."""
        if self.path and os.path.exists(self.path):
            self.logger.debug(f"Removing temporary file {self.path} ...")
            os.remove(self.path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

def load_filters() -> dict[str: ColorFilter]:

    base_dir = Path(__file__).parent 
    toml_path = base_dir / 'filters.toml'

    default_filters = {
        "Orange": ((252, 176, 32), (10, 6, 3)),
        "Capuccino": ((200, 185, 150), (61, 49, 40)),
        "Brat": ((137, 205, 0), (0, 0, 0)),
        "Fairy": ((174, 255, 223), (90, 84, 117)),
        "Bloody": ((255, 42, 0), (43, 12, 0)),
        "Lavender": ((196, 167, 231), (35, 33, 54)),
        "Cyan": ((0, 204, 255), (0, 34, 43)),
        "Vapor": ((250, 185, 253), (75, 123, 222)),
        "Matrix": ((0, 255, 0), (0, 39, 6)),
        "ObraDinn": ((229, 255, 254), (51, 51, 25))
    }

    try:
        with open(toml_path, 'rb') as file:
            filters = tomllib.load(file)
        return filters
    
    except FileNotFoundError:
        print("[FileNotFoundError] The 'filters.toml' file was not found. Using default filter pack...")
        return default_filters
    
    except tomllib.TOMLDecodeError as e:
        print(f"[TOMLDecodeError] Error parsing the TOML file: {e}\n Using default filter pack...")
        return default_filters