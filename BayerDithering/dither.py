import numpy as np
from numpy.typing import NDArray
import cv2
import imageio
import time
import os
import io
import contextlib
import logging
import tempfile
from .core import DitherConfig, MediaProcessor, MediaInput, MediaOutput
from .gpu import GPUProcessor, TAICHI_AVAILABLE
from .cpu import CPUProcessor
from .utils import ProcessedGIF, ProcessedVideo


class BayerDither():
    """Main orchestrator for the Bayer Dithering library.
    
    Handles media routing (images vs. video) and delegates the actual 
    computation to the injected processor (CPU or GPU).
    """
    def __init__(self, processor: MediaProcessor, verbose: bool = False) -> None:
        """Initializes the Dithering orchestrator.

        Args:
            processor (MediaProcessor): An initialized instance of CPUProcessor or GPUProcessor.
            verbose (bool, optional): Enables detailed DEBUG logging. Defaults to False.
        """

        self.logger = logging.getLogger(__name__)
        level = logging.DEBUG if verbose else logging.WARNING
        self.logger.setLevel(level)
        
        self.processor = processor
        self.config = self.processor.config

        self.device = "gpu" if isinstance(processor, GPUProcessor) else "cpu"

    def _get_video_metadata(self, video: cv2.VideoCapture) -> tuple[int, int, int, float]:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            self.logger.warning("FPS invalid or not detected, defaulting to 30")
            fps = 30

        self.logger.debug(
            f"Video info: {width}x{height} @ {fps} FPS | total_frames={total_frames}"
        )

        return width, height, total_frames, fps

    def _create_writer(self, width, height, fps) -> tuple[cv2.VideoWriter, str]:
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        is_color = self.processor.config.filter is not None
        if isinstance(self.processor, GPUProcessor):
            is_color = True

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            filename=temp_path,
            fourcc=fourcc,
            fps=fps,
            frameSize=(width, height),
            isColor=is_color
        )

        return out, temp_path

    def _compute_output_size(self, width: int, height: int) -> tuple[int, int]:
        if (self.processor.config.downscale_factor > 1 and not self.processor.config.upscale):

            new_width = max(1, width // self.processor.config.downscale_factor)
            new_height = max(1, height // self.processor.config.downscale_factor)

            self.logger.debug(
                f"Downscaling output to {new_width}x{new_height} "
                f"(factor={self.processor.config.downscale_factor})"
            )

            return new_width, new_height

        return width, height

    def _apply_to_video_cpu(self, video: cv2.VideoCapture) -> ProcessedVideo:
        processor: CPUProcessor = self.processor
        start: float = time.time()

        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        width, height, total_frames, fps = self._get_video_metadata(video)
        width, height = self._compute_output_size(width, height)
        
        out, temp_path = self._create_writer(width, height, fps)

        frame_counter: int = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            processed: NDArray[np.uint8] = processor.process_frame(frame)
            out.write(processed)
            frame_counter += 1

            if frame_counter % 100 == 0 or frame_counter == total_frames:
                self.logger.debug(f"Writer progress: {frame_counter}/{total_frames}")

        out.release()

        elapsed: float = time.time() - start
        fps_eff: float = total_frames / elapsed if elapsed > 0 else 0

        self.logger.info("Video processing completed")
        self.logger.info(f"Processing completed in {elapsed:.2f}s | Effective FPS: {fps_eff:.2f}")
        self.logger.debug(f"Temporary file created at: {temp_path}")

        return ProcessedVideo(temp_path, self.logger.getEffectiveLevel())

    def _apply_to_video_gpu(self, video: cv2.VideoCapture) -> ProcessedVideo:
        processor: GPUProcessor = self.processor
        start: float = time.time()
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        width, height, total_frames, fps = self._get_video_metadata(video)
        
        downscaled_w = max(1, width // self.config.downscale_factor)
        downscaled_h = max(1, height // self.config.downscale_factor)

        writer_width: int = width if self.config.upscale else downscaled_w
        writer_height: int = height if self.config.upscale else downscaled_h

        out, temp_path = self._create_writer(writer_width, writer_height, fps)

        batch_size: int = 128
        frame_counter: int = 0
        batch: list[NDArray[np.uint8]] = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if len(batch) < batch_size:
                batch.append(frame)
            else:
                result_batch = processor.process_video_batch(batch)

                for processed_frame in result_batch:
                    if self.config.upscale:
                        processed_frame = cv2.resize(
                            processed_frame,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST
                        )
                    out.write(processed_frame)
                    frame_counter += 1

                    if frame_counter % 100 == 0 or frame_counter == total_frames:
                        self.logger.debug(f"Writer progress: {frame_counter}/{total_frames}")

                batch.clear()
                batch.append(frame)

        if batch:
            result_batch = processor.process_video_batch(batch)

            for processed_frame in result_batch:
                if self.config.upscale:
                    processed_frame = cv2.resize(
                        processed_frame,
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
                out.write(processed_frame)
                frame_counter += 1

                if frame_counter % 100 == 0 or frame_counter == total_frames:
                    self.logger.debug(f"Writer progress: {frame_counter}/{total_frames}")

            batch.clear()

        out.release()

        elapsed: float = time.time() - start
        fps_eff: float = total_frames / elapsed if elapsed > 0 else 0

        self.logger.info("Video processing completed")
        self.logger.info(f"Processing completed in {elapsed:.2f}s | Effective FPS: {fps_eff:.2f}")
        self.logger.debug(f"Temporary file created at: {temp_path}")

        return ProcessedVideo(temp_path, self.logger.getEffectiveLevel())
            
    def apply_to_video(self, video: cv2.VideoCapture) -> ProcessedVideo:
        """Applies the dithering process to an entire video stream.

        Args:
            video (cv2.VideoCapture): An opened OpenCV VideoCapture object.

        Returns:
            ProcessedVideo: A wrapper object containing the path to the processed temporary video file.
        """

        config: DitherConfig = self.config

        self.logger.info(
            f"Starting video processing | "
            f"device={self.device} | "
            f"matrix={config.b_matrix.shape} | "
            f"contrast={config.contrast} | "
            f"sharpness={config.sharpness} | "
            f"downscale={config.downscale_factor} | "
            f"upscale={config.upscale} | "
            f"filter={'yes' if config.filter else 'no'}"
        )

        if self.device == "gpu":
            result = self._apply_to_video_gpu(video)
        else:
            result = self._apply_to_video_cpu(video)

        return result

    def apply_to_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Applies the dithering process to a single image array.

        Args:
            image (NDArray[np.uint8]): The source image as a NumPy array (BGR format).

        Returns:
            NDArray[np.uint8]: The processed image as a NumPy array.
        """

        config: DitherConfig = self.config

        self.logger.info(
            f"Starting image processing | "
            f"device={self.device} | "
            f"matrix={config.b_matrix.shape} | "
            f"contrast={config.contrast} | "
            f"sharpness={config.sharpness} | "
            f"downscale={config.downscale_factor} | "
            f"upscale={config.upscale} | "
            f"filter={'yes' if config.filter else 'no'}"
        )

        size_mb: float = image.nbytes / (1024 * 1024)
        h, w = image.shape[:2]
        channels: int = 1 if len(image.shape) == 2 else image.shape[2]

        self.logger.info(f"Image info: "
            f"resolution={w}x{h} | "
            f"channels={channels} | "
            f"size={size_mb:.2f}MB | " 
            f"dtype={image.dtype}"
        )

        start: float = time.time()
        
        result: NDArray[np.uint8] = self.processor.process_frame(image)

        elapsed : float = time.time() - start
        self.logger.debug(f"Processing completed in {elapsed:.2f}s")
        
        return result

    def apply_to_gif(self, gif_reader: imageio.v2.LegacyReader) -> ProcessedGIF:
        """Applies the dithering process to an animated GIF stream.

        Args:
            gif_reader: An opened imageio Reader object.

        Returns:
            ProcessedGIF: A wrapper object containing the path to the temporary processed GIF.
        """
        
        config: DitherConfig = self.config

        self.logger.info(
            f"Starting GIF processing | "
            f"device={self.device} | " 
            f"matrix={config.b_matrix.shape} | "
            f"contrast={config.contrast} | "
            f"sharpness={config.sharpness} | "
            f"downscale={config.downscale_factor} | "
            f"upscale={config.upscale} | "
            f"filter={'yes' if config.filter else 'no'}"
        )

        start = time.time()

        meta = gif_reader.get_meta_data()
        duration = meta.get('duration', 100) # ms per frame
        loop = meta.get('loop', 0)
        
        processed_frames = []
        batch = []
        batch_size = 128 if self.device == "gpu" else 1

        for frame in gif_reader:
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            if self.device == "cpu":
                proc = self._apply_to_video_cpu_frame(frame)
                processed_frames.append(proc)
            else:
                batch.append(frame)
                if len(batch) == batch_size:
                    res_batch = self.processor.process_video_batch(batch)
                    for idx, r in enumerate(res_batch):
                        proc = self._post_process_frame(r, batch[idx].shape[:2])
                        processed_frames.append(proc)
                    batch.clear()

        if batch and self.device == "gpu":
            res_batch = self.processor.process_video_batch(batch)
            for idx, r in enumerate(res_batch):
                proc = self._post_process_frame(r, batch[idx].shape[:2])
                processed_frames.append(proc)

        fd, temp_path = tempfile.mkstemp(suffix=".gif")
        os.close(fd)
        
        self.logger.debug("Writing GIF frames to temporary file...")
        with imageio.get_writer(temp_path, format='GIF', duration=duration, loop=loop) as writer:
            for f in processed_frames:
                writer.append_data(f)

        elapsed = time.time() - start
        self.logger.info(f"GIF processing completed in {elapsed:.2f}s")
        
        return ProcessedGIF(temp_path, self.logger.getEffectiveLevel())

    def _apply_to_video_cpu_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Helper to process a frame on CPU and convert back to RGB."""
        original_shape = frame.shape[:2]
        proc = self.processor.process_frame(frame)
        return self._post_process_frame(proc, original_shape)

    def _post_process_frame(self, frame: NDArray[np.uint8], original_shape: tuple[int, int]) -> NDArray[np.uint8]:
        """Helper to handle upscaling and BGR->RGB conversion for GIFs."""
        if self.config.upscale:
            frame = cv2.resize(frame, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def apply(self, media: MediaInput) -> MediaOutput:
        """Unified method to apply the dithering effect to supported media streams.

        Args:
            media (MediaInput): cv2.VideoCapture, NDArray (image), or imageio.Reader (GIF).

        Returns:
            MediaOutput: The processed result (NDArray, ProcessedVideo, or ProcessedGIF).
            
        Raises:
            TypeError: If the media type is not supported.
        """

        if isinstance(media, cv2.VideoCapture):
            return self.apply_to_video(video=media)
        elif type(media) is np.ndarray:
            return self.apply_to_image(image=media)
        elif hasattr(media, 'get_meta_data') and hasattr(media, 'get_length'):
            return self.apply_to_gif(gif_reader=media)
        else:
            raise TypeError(
                f"Unsupported media type: {type(media)}"
            )

    @staticmethod
    def get_available_devices() -> list[str]:
        """Checks the system for supported processing hardware.

        Returns:
            list[str]: A list of available devices. Will always include 'cpu', 
                       and will include 'gpu' if Taichi detects a valid backend (CUDA, Vulkan, or Metal).
        """

        devices = ["cpu"]

        if not TAICHI_AVAILABLE:
            return devices
        
        
        # temporarily mute Taichi to prevent it from printing errors 
        # if the user doesn't have a GPU drivers installed.
        os.environ["TI_LOG_LEVEL"] = "ERROR"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import taichi as ti
            try:
                # list of backend enums it can use on this machine
                archs = ti.supported_archs()
                
                # Check if there is any hardware accelerator beyond the CPU
                if any(arch in archs for arch in [ti.cuda, ti.vulkan, ti.metal, ti.opengl, ti.dx11]):
                    devices.append("gpu")
            except Exception:
                pass
                
        return devices
