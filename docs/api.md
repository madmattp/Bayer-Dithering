# API Reference

Welcome to the complete API reference for BayerDithering.

## Types & Aliases

These custom type aliases are used throughout the library to ensure static typing safety and make function signatures cleaner.

::: BayerDithering.core.MediaInput

::: BayerDithering.core.MediaOutput

::: BayerDithering.core.RGBColor

::: BayerDithering.core.ColorFilter

## Core Processing

Below are the primary classes responsible for orchestrating the dithering process and configuring the media pipeline.

::: BayerDithering.DitherConfig

::: BayerDithering.BayerDither

## Hardware Backends

The library delegates the heavy computational lifting to these `MediaProcessor` classes. The GPUProcessor leverages the Taichi framework for massive parallel execution, while the CPUProcessor uses optimized NumPy and OpenCV routines as a fallback.

::: BayerDithering.core.MediaProcessor

::: BayerDithering.CPUProcessor

::: BayerDithering.GPUProcessor

## Utilities & Media Wrappers

Helper classes used internally to securely manage temporary file streams, merge audio, and ensure proper cleanup from the disk after the processing is finished.

::: BayerDithering.utils.ProcessedVideo

::: BayerDithering.utils.ProcessedGIF

::: BayerDithering.utils.load_filters

## Matrix Generation & Presets

Here you can find the dynamic matrix generator and the pre-computed matrix presets available for immediate use.

::: BayerDithering.matrices

::: BayerDithering.generate_bayer_matrix
