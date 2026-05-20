from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, TypeAlias, Any, Union
import numpy as np
from numpy.typing import NDArray
import cv2
from .utils import ProcessedVideo, ProcessedGIF


MediaInput: TypeAlias = Union[NDArray[np.uint8], cv2.VideoCapture, Any]
MediaOutput: TypeAlias = Union[NDArray[np.uint8], ProcessedVideo, ProcessedGIF]

RGBColor: TypeAlias = tuple[int, int, int]
ColorFilter: TypeAlias = tuple[RGBColor, RGBColor]

@dataclass
class DitherConfig:
    """Main configuration for the Bayer Dithering process.

    Attributes:
        b_matrix (NDArray[np.float32]): Normalized Bayer matrix (values from 0.0 to 1.0).
        contrast (float): Contrast multiplier applied before dithering (default: 1.5).
        sharpness (float): Intensity of the unsharp mask filter (default: 1.6).
        downscale_factor (int): Factor by which the input is downscaled before processing (default: 1).
        upscale (bool): If True, resizes the processed image back to its original dimensions (default: False).
        filter (Optional[ColorFilter]): Tuple with two RGB colors (light, dark) to apply as a duotone palette (default: None).
    """

    b_matrix: NDArray[np.float32]
    contrast: float = 1.5
    sharpness: float = 1.6
    downscale_factor: int = 1
    upscale: bool = False
    filter: Optional[ColorFilter] = None

class MediaProcessor(ABC):
    """Base interface for all media processors (CPU and GPU)."""
    def __init__(self, config: DitherConfig) -> None:
        self.config = config

    @abstractmethod
    def process_frame(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        raise NotImplementedError