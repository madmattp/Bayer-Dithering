# Quickstart (API Usage)

This guide will show you how to integrate `BayerDithering` into your Python scripts to process images/videos programmatically.

## 1. Basic Image Processing (NumPy / OpenCV)

If you already have an image loaded as a NumPy array, you can process it directly:

```python
import cv2
from BayerDithering import BayerDither, CPUProcessor, DitherConfig, matrices

# Create the pipeline configuration
config = DitherConfig(
    b_matrix=matrices["4x4"],
    contrast=1.5,
    sharpness=1.6,
    downscale_factor=2,
    upscale=True,
    filter=None  # Pass a tuple of RGB colors or None for grayscale
)

# Initialize the processor (CPU or GPUProcessor) and the ditherer
processor = CPUProcessor(config)
ditherer = BayerDither(processor=processor, verbose=True)

# Apply dithering to a NumPy array (BGR image from OpenCV)
image = cv2.imread("media/cat.jpg")
dithered_image = ditherer.apply(image)

# Save the result
cv2.imwrite("media/cat_dithered.png", dithered_image)
```

## 2. High-Performance Video Processing (GPU Accelerated)

For processing videos, pass a `cv2.VideoCapture` object directly into the `apply` method. The pipeline leverages the `taichi` backend for parallel GPU frame processing and returns a `ProcessedVideo` context manager:

```python
import cv2
from BayerDithering import BayerDither, GPUProcessor, DitherConfig, matrices
from BayerDithering.utils import ProcessedVideo

config = DitherConfig(
    b_matrix=matrices["8x8"],
    contrast=1.3,
    sharpness=1.5,
    downscale_factor=2
)

# Use GPUProcessor for hardware acceleration
processor = GPUProcessor(config)
ditherer = BayerDither(processor=processor)

# Open the video stream using OpenCV
video_capture = cv2.VideoCapture("media/huh.mp4")

# Pass the video capture object to the ditherer.
# The router will automatically return a ProcessedVideo context.
# Always use 'with' to ensure secure handling and cleanup of temporary files.
with ditherer.apply(video_capture) as result:
    result.save_with_audio(original_video_path="media/huh.mp4", path="media/huh_dithered.mp4")

# Remember to release the video hardware resource
video_capture.release()
```

## 3. Animated GIF Processing

GIF processing relies on multi-frame arrays or lists of images. The `apply` method processes the frame sequence and returns a `ProcessedGIF` context manager to gracefully handle saving the output stream:

```python
import imageio as iio
from BayerDithering import BayerDither, CPUProcessor, DitherConfig, matrices
from BayerDithering.utils import ProcessedGIF

config = DitherConfig(
    b_matrix=matrices["4x4"],
    contrast=1.6,
    sharpness=1.4,
    downscale_factor=3,
    upscale=True
)

# Initialize using CPU or GPU (both support GIF frame-by-frame processing)
processor = CPUProcessor(config)
ditherer = BayerDither(processor=processor)

# Load the GIF frames as a single multi-frame NumPy array using imageio
with iio.get_reader("media/cat-shocked.gif") as gif_frames:

  # Pass the frame sequence object to the ditherer. 
  # The router will return a ProcessedGIF instance
  with ditherer.apply(gif_frames) as result:
    result.save(dest_path="media/cat_shocked_dithered.gif")
```

## 4. Loading Custom Filters Programmatically

If you want to use the color palettes defined in your [`filters.toml`](https://github.com/madmattp/Bayer-Dithering/blob/7c007c2f3dce9cce1d17fbcee1e92f132920934b/BayerDithering/filters.toml) dynamically inside a Python script:

```python
from BayerDithering.utils import load_filters
from BayerDithering import DitherConfig, matrices

# Load all filters as a dictionary
filters = load_filters()

# Extract the RGB data for a specific palette (e.g., 'Cyan')
cyan_palette = filters.get("Cyan")

config = DitherConfig(
    b_matrix=matrices["4x4"],
    filter=cyan_palette  # Pass the loaded palette data to the configuration
)
```
