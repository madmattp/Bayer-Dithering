# Command Line Interface (CLI)

When you install `BayerDithering`, it automatically adds the global `dither` command to your system's terminal. This tool allows you to process images, GIFs, and videos without writing a single line of Python code.

## Basic Usage

The only strictly required argument is the input file (`-i` or `--input`). The tool will automatically detect the media type (image, video, or GIF) and generate an output file in the same directory.

```bash
dither -i my_photo.jpg
```

*This will create* `my_photo_dithered.png` *using the default settings (CPU, 4x4 matrix, downscale factor 2, etc).*

## Available Options

Here is the complete list of flags you can use to customize the dithering effect:

### Core Arguments

* **`-i, --input PATH`** *(Required)*: Path to the input file (supports `.jpg`, `.png`, `.mp4`, `.gif`, etc).

* **`-o, --output PATH`**: Specifies the output file path. If omitted, the tool generates a smart default name based on the input format.

* **`-a, --arch {cpu, gpu}`**: Selects the hardware backend. Defaults to `cpu`. *Note: `gpu` requires the `taichi` backend to be installed.*

### Visual Adjustments

* **`-m, --matrix {2x2, 4x4, 8x8}`**: The Bayer matrix size. Larger matrices create more color gradations but larger patterns. Defaults to `4x4`.

* **`-d, --downscale FACTOR`**: Integer factor to shrink the image before dithering (e.g., `2` cuts resolution in half). Great for a retro, chunky pixel look. Defaults to `2`.

* **`-u, --upscale BOOLEAN`**: If `True`, resizes the image back to its original dimensions after processing. Defaults to `True`.

* **`-c, --contrast FACTOR`**: Float multiplier to adjust contrast. Defaults to `1.5`.

* **`-s, --sharpness FACTOR`**: Intensity of the unsharp mask filter applied before dithering. Defaults to `1.6`.

### Styling & Output

* **`-f, --filter FILTER`**: Applies a custom duotone palette. Available options: `Orange`, `Capuccino`, `Brat`, `Fairy`, `Bloody`, `Lavender`, `Cyan`, `Vapor`, `Matrix`, `ObraDinn`, `Bill`, `Gruvbox`.

* **`-q, --quiet`**: Suppresses all terminal output and progress logs.

## Practical Examples

### 1. High-Speed Video Processing

Processing a video using the GPU backend for maximum performance, using an 8x8 matrix for smoother gradients:

```bash
dither -i gameplay.mp4 -a gpu -m 8x8
```

### 2. The "Vaporwave" Aesthetic

Applying the Vaporwave color filter with increased contrast to a photo:

```bash
dither -i portrait.png -f Vapor -c 2.0
```

### 3. Pure Retro Pixel-Art

Applying a heavy downscale (4x) and disabling the upscale to output a tiny, raw pixel-art image:

```bash
dither -i landscape.jpg -d 4 -u false
```

### 4. GIF Processing

Applying the 'Orange' filter to an animated GIF quietly (no logs):

```bash
dither -i cat_meme.gif -f Orange -q
```
