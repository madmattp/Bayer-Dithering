# Bayer Dithering

A high-performance, GPU-accelerated Python library and Command-Line Interface (CLI) that applies Bayer matrix dithering to images, GIFs, and videos.

It offers a variety of customizable options such as matrix size, custom color filters, sharpness, contrast, and downscaling, powered by parallel processing for blazing-fast media generation.

## Features

- **Hardware Acceleration:** Choose between CPU or GPU (`taichi` backend) for massive performance gains, especially on videos.
- **Universal Media Support:** Seamlessly process PNG, JPG, GIF, and MP4 files with automatic format detection.
- **Global CLI:** Install once and use the `dither` command from anywhere in your terminal.
- **Customizable Filters:** Apply beautiful retro color palettes easily.
- **Pre-Processing Pipeline:** Built-in options for downscaling, contrast adjustment, and sharpening before the dithering effect is applied.

## Documentation Sections

Get started by exploring the sections below:

- **[Installation Guide](installation.md):** Learn how to install the library and configure the GPU backend.
- **[Quickstart (API Usage) with Examples](quickstart.md):** Discover how to integrate `BayerDithering` into your Python scripts, process images programmatically, and easily customize contrast, sharpness, and color palettes.
- **[Command Line Interface](cli.md):** Discover how to use the global `dither` command to process files via terminal.
- **[Gallery & Examples](examples.md):** See the visual results of our pre-configured color filters and matrix sizes.
- **[API Reference](api.md):** Detailed technical documentation for modules, classes, and Taichi kernels.
