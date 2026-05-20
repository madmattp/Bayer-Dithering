import io
import contextlib
import os
import cv2
import numpy as np
from numpy.typing import NDArray
import taichi as ti
from .core import MediaProcessor, DitherConfig


# Suppress Taichi's hardcoded Python prints during import and internal C++ engine logs
os.environ["TI_LOG_LEVEL"] = "error"
with contextlib.redirect_stdout(io.StringIO()):
    import taichi as ti


@ti.data_oriented
class GPUProcessor(MediaProcessor):
    """GPU-accelerated implementation for Bayer Dithering using Taichi.
    
    Highly recommended for video processing due to its massive parallelization capabilities.
    """

    def __init__(self, config: DitherConfig) -> None:
        super().__init__(config)

        with contextlib.redirect_stdout(io.StringIO()):
            ti.init(arch=ti.gpu, verbose=False)

        self.batch_size = None
        self.height = None
        self.width = None
        self.out_height = None
        self.out_width = None

        self.frames = None
        self.out_frames = None
        
        self.bayer_size = self.config.b_matrix.shape
        self.bayer_h, self.bayer_w = self.bayer_size
        
        self.bayer = ti.field(dtype=ti.f32, shape=self.bayer_size)
        self.bayer.from_numpy(config.b_matrix.astype(np.float32))

    def _set_frames_buffer(self, shape: tuple[int, int], new_shape: tuple[int, int], batch_size: int = 64) -> None:
        """Initializes or resizes the Taichi ndarrays used for GPU memory buffering.

        Args:
            shape (tuple[int, int]): Original (height, width) of the frames.
            new_shape (tuple[int, int]): Target (height, width) after downscaling.
            batch_size (int, optional): Number of frames to process in a single batch. Defaults to 64.
        """

        h, w = shape
        h1, w1 = new_shape

        self.batch_size = batch_size
        self.height = h
        self.width = w
        self.out_height = h1
        self.out_width = w1

        self.frames = ti.ndarray(dtype=ti.u8, shape=(batch_size, h, w, 3))
        self.out_frames = ti.ndarray(dtype=ti.u8, shape=(batch_size, h1, w1, 3))

    @ti.func
    def get_gray(self, frames: ti.template(), n: ti.i32, y: ti.i32, x: ti.i32) -> ti.f32:
        """Taichi sub-function to extract and convert a specific BGR pixel to grayscale luminance."""

        b = ti.cast(frames[n, y, x, 0], ti.f32)
        g = ti.cast(frames[n, y, x, 1], ti.f32)
        r = ti.cast(frames[n, y, x, 2], ti.f32)
        return 0.299 * r + 0.587 * g + 0.114 * b

    @ti.kernel
    def compute_batch_mean(
        self,
        frames: ti.types.ndarray(dtype=ti.u8, ndim=4),
        batch_len: ti.i32,
        src_h: ti.i32,
        src_w: ti.i32
    ) -> ti.f32:
        """Parallel reduction kernel to compute the mean luminance of an entire video batch.

        Returns:
            ti.f32: The overall mean grayscale value of the batch.
        """

        sum_gray = 0.0
        for n, y, x in ti.ndrange(batch_len, src_h, src_w):
            sum_gray += self.get_gray(frames, n, y, x)
            
        total_pixels = ti.cast(batch_len * src_h * src_w, ti.f32)
        return sum_gray / total_pixels

    @ti.kernel
    def process_video_batch_kernel(
        self,
        frames: ti.types.ndarray(dtype=ti.u8, ndim=4),
        out_frames: ti.types.ndarray(dtype=ti.u8, ndim=4),
        batch_len: ti.i32,
        src_h: ti.i32,
        src_w: ti.i32,
        out_h: ti.i32,
        out_w: ti.i32,
        downscale: ti.i32,
        contrast: ti.f32,
        mean_val: ti.f32,
        sharpness: ti.f32,
        use_filter: ti.i32,
        r0: ti.u8, g0: ti.u8, b0: ti.u8,
        r1: ti.u8, g1: ti.u8, b1: ti.u8
    ):
        """Core GPU kernel that applies unsharp masking, contrast adjustment, and Bayer dithering to a batch of frames."""

        for n, y, x in ti.ndrange(batch_len, out_h, out_w):
            src_x = x * downscale
            src_y = y * downscale

            # Sharpness via Convolução 3x3 (lendo os vizinhos)
            x_left = ti.max(0, src_x - 1)
            x_right = ti.min(src_w - 1, src_x + 1)
            y_up = ti.max(0, src_y - 1)
            y_down = ti.min(src_h - 1, src_y + 1)

            center = self.get_gray(frames, n, src_y, src_x)
            up = self.get_gray(frames, n, y_up, src_x)
            down = self.get_gray(frames, n, y_down, src_x)
            left = self.get_gray(frames, n, src_y, x_left)
            right = self.get_gray(frames, n, src_y, x_right)

            # Aplica matriz laplaciana para criar a nitidez nas bordas
            gray = center + sharpness * (4.0 * center - up - down - left - right)

            # Contrast
            gray = (gray - mean_val) * contrast + mean_val

            # Clamp (limita os resultados matemáticos às cores visíveis de 0 a 255)
            gray = ti.max(0.0, ti.min(255.0, gray))

            # Bayer Dither
            by = y % self.bayer_h
            bx = x % self.bayer_w
            threshold = self.bayer[by, bx] * 255.0
            val = 255 if gray > threshold else 0

            if use_filter == 1:
                if val == 0:
                    out_frames[n, y, x, 0] = b0
                    out_frames[n, y, x, 1] = g0
                    out_frames[n, y, x, 2] = r0
                else:
                    out_frames[n, y, x, 0] = b1
                    out_frames[n, y, x, 1] = g1
                    out_frames[n, y, x, 2] = r1
            else:
                val_u8 = ti.cast(val, ti.u8)
                out_frames[n, y, x, 0] = val_u8
                out_frames[n, y, x, 1] = val_u8
                out_frames[n, y, x, 2] = val_u8
    
    def process_video_batch(self, batch: list[NDArray[np.uint8]]) -> NDArray[np.uint8]:
        """Manages the memory transfer and execution of the GPU kernel for a batch of frames.

        Args:
            batch (list[NDArray[np.uint8]]): A list of raw BGR frames.

        Returns:
            NDArray[np.uint8]: A numpy array containing the processed frames.
        """

        if not batch:
            return []

        original_h, original_w = batch[0].shape[:2]
        current_batch_len = len(batch)

        downscaled_w = max(1, original_w // self.config.downscale_factor)
        downscaled_h = max(1, original_h // self.config.downscale_factor)

        if (self.frames is None or 
            self.height != original_h or 
            self.width != original_w or 
            self.batch_size < current_batch_len):
            
            self._set_frames_buffer(
                shape=(original_h, original_w),
                new_shape=(downscaled_h, downscaled_w),
                batch_size=current_batch_len
            )

        batch_np = np.ascontiguousarray(np.stack(batch), dtype=np.uint8)

        temp = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.uint8)
        temp[:current_batch_len] = batch_np

        self.frames.from_numpy(temp)

        r0 = g0 = b0 = r1 = g1 = b1 = 0
        use_filter = 0

        if self.config.filter:
            (r1, g1, b1), (r0, g0, b0) = self.config.filter
            use_filter = 1

        mean_val = self.compute_batch_mean(
            self.frames, current_batch_len, self.height, self.width
        )

        self.process_video_batch_kernel(
            self.frames,
            self.out_frames,
            current_batch_len,
            self.height,
            self.width,
            self.out_height,
            self.out_width,
            self.config.downscale_factor,
            self.config.contrast,
            mean_val,
            self.config.sharpness,
            use_filter,
            r0, g0, b0,
            r1, g1, b1
        )

        result = self.out_frames.to_numpy()
        return result[:current_batch_len]

    def process_frame(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Executes the GPU pipeline for a single frame.
        
        Args:
            image (NDArray[np.uint8]): Input image in BGR format.

        Returns:
            NDArray[np.uint8]: Processed dithered image.
        """

        original_h, original_w = image.shape[:2]

        downscaled_w = max(1, original_w // self.config.downscale_factor)
        downscaled_h = max(1, original_h // self.config.downscale_factor)

        if (self.frames is None or 
            self.height != original_h or 
            self.width != original_w or 
            self.batch_size != 1):
            
            self._set_frames_buffer(
                shape=(original_h, original_w),
                new_shape=(downscaled_h, downscaled_w),
                batch_size=1
            )

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        temp = np.zeros((1, self.height, self.width, 3), dtype=np.uint8)
        temp[0] = np.ascontiguousarray(image)
        self.frames.from_numpy(temp)

        r0 = g0 = b0 = r1 = g1 = b1 = 0
        use_filter = 0
        if self.config.filter:
            (r1, g1, b1), (r0, g0, b0) = self.config.filter
            use_filter = 1

        mean_val = self.compute_batch_mean(self.frames, 1, self.height, self.width)

        self.process_video_batch_kernel(
            self.frames,
            self.out_frames,
            1,
            self.height,
            self.width,
            self.out_height,
            self.out_width,
            self.config.downscale_factor,
            self.config.contrast,
            mean_val,
            self.config.sharpness,
            use_filter,
            r0, g0, b0,
            r1, g1, b1
        )

        result = self.out_frames.to_numpy()[0]

        if self.config.upscale:
            result = cv2.resize(
                result, 
                (original_w, original_h), 
                interpolation=cv2.INTER_NEAREST
            )

        return result
    
