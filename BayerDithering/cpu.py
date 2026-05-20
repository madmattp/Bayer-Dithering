import logging
import cv2
import numpy as np
from .core import MediaProcessor, DitherConfig, ColorFilter
from numpy.typing import NDArray


class CPUProcessor(MediaProcessor):
    """Pure CPU implementation for Bayer Dithering.
    
    Ideal for processing single images or when hardware acceleration (GPU) is unavailable.
    """

    def __init__(self, config: DitherConfig) -> None:
        super().__init__(config)

        self.bayer_matrix_u8 = (self.config.b_matrix * 255).astype(np.uint8)

    @staticmethod
    def apply_color_filter(image: NDArray[np.uint8], colors: ColorFilter) -> NDArray[np.uint8]:
        """Applies a duotone filter to a binarized image.

        Args:
            image (NDArray[np.uint8]): Binarized grayscale image (values 0 and 255 only).
            colors (ColorFilter): A tuple containing (light_color_rgb, dark_color_rgb).

        Returns:
            NDArray[np.uint8]: Colored image in BGR format.
        """

        light, dark = colors  # (R,G,B), (R,G,B)

        # Convert RGB to BGR for OpenCV compatibility
        light = light[::-1]
        dark = dark[::-1]

        h, w = image.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        mask = image == 255  # white
        colored[mask] = light
        colored[~mask] = dark

        return colored

    @staticmethod
    def upscale(image: NDArray[np.uint8], target_size: tuple[int, int]) -> NDArray[np.uint8]:
        """Resizes the image to a target size using nearest-neighbor interpolation."""

        return cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)

    def downscale(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Reduces the image resolution based on the configured downscale_factor."""

        height, width = image.shape[:2]

        new_width = max(1, width // self.config.downscale_factor)
        new_height = max(1, height // self.config.downscale_factor)
        new_size = (new_width, new_height)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    
    def adjust_contrast(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Adjusts the contrast of an image based on its mean luminance."""

        mean = np.mean(image)
        return np.clip(
            (image - mean) * self.config.contrast + mean,
            0, 255
        ).astype(np.uint8)

    def sharpen(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Applies an unsharp mask filter to enhance edges."""

        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(image, 1 + self.config.sharpness, blurred, -self.config.sharpness, 0)

    def bayer_dither_array(self, image_array: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Applies the Bayer thresholding to a grayscale image array."""

        matrix = self.bayer_matrix_u8

        h, w = image_array.shape
        m = matrix.shape[0]

        tiled_matrix = np.tile(matrix, (h // m + 1, w // m + 1))[:h, :w]

        return (image_array > tiled_matrix).astype(np.uint8) * 255

    def process_frame(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Executes the full CPU processing pipeline on a single frame."""

        if image is None:
            raise ValueError("Invalid input image provided.")
        
        original_h, original_w = image.shape[:2]

        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = self.adjust_contrast(image=grayscale_img)
        img = self.sharpen(image=img)
        img = self.downscale(image=img)
        img = self.bayer_dither_array(img)

        if self.config.filter is not None:
            img = self.apply_color_filter(img, self.config.filter)
        
        if self.config.upscale:
            img = self.upscale(image=img, target_size=(original_w, original_h))

        return img
