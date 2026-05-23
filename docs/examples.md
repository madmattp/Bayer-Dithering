# Gallery & Examples

This page serves as a visual showcase for `BayerDithering`. Below you will find comparisons of different matrix sizes, parameter adjustments, and the built-in color filters applied to images and animations.

---

## Matrix Size Comparison

The size of the Bayer matrix significantly changes the texture and granularity of the final dithered image. Larger matrices create smoother gradients with more perceived shading, while smaller matrices result in a high-contrast, sharper pixel-art style.

| Original | 2x2 Matrix | 4x4 Matrix (Default) | 8x8 Matrix |
| :---: | :---: | :---: | :---: |
| ![Original](media\cat.jpg) | ![2x2 Dither](media\cat_dithered_2x2.png) | ![4x4 Dither](media\cat_dithered_4x4.png) | ![8x8 Dither](media\cat_dithered_8x8.png) |

*Tip: Use the `-m` or `--matrix` flag in the CLI to switch between presets.*

---

## Fine-Tuning Parameters

Before thresholding the pixels against the Bayer matrix, the library pre-processes the frame using contrast and sharpness filters to enhance edge definition.

### Sharpness Configuration

Adjusting the unsharp mask filter via `--sharpness` ensures that fine details are not lost during downscaling.

* **Low Sharpness (`-s 0.5`):** Smoother, softer blends. Good for low-contrast portraits.
* **Default Sharpness (`-s 1.6`):** Balanced edge enhancement.
* **High Sharpness (`-s 3.0`):** Heavy outlines, perfect for cartoonish or architectural styles.

### Contrast Configuration

Contrast (`--contrast`) acts as a pivot around the mean luminance of the frame, amplifying light and dark sections.

* **Low Contrast (`-c 1.0`):** Flatter look, preserves more mid-tones.
* **Default Contrast (`-c 1.5`):** Striking retro gradients.
* **High Contrast (`-c 2.5`):** Extreme black-and-white silhouettes, eliminating subtle gradients.

---

## Built-in Color Filters (Duotone Packs)

The library features several pre-configured duotone color palettes designed to give your media a unique retro or cinematic mood. You can apply them using the `-f` or `--filter` flag.

### Popular Aesthetic Packs

#### 📟 Matrix (`-f Matrix`)

The classic digital rain aesthetic featuring vibrant green over deep black.
![Matrix Filter Example](media\cat_dithered_matrix.png){ width="70%" }

#### 🍊 Orange (`-f Orange`)

A warm, sunset-inspired palette that brings out rich golden and deep brown tones.
![Orange Filter Example](media\cat_dithered_orange.png){ width="70%" }

#### 🧚 Fairy (`-f Fairy`)

A mystical combination of soft mint greens and muted lavender/purple shadows.
![Fairy Filter Example](media\cat_dithered_fairy.png){ width="70%" }

#### 🪵 Gruvbox (`-f Gruvbox`)

A retro, warm, pastel-tinted palette that is easy on the eyes.
![Gruvbox Filter Example](media\cat_dithered_gruvbox.png){ width="70%" }

#### 🩸 Bloody (`-f Bloody`)

An intense, high-contrast palette featuring aggressive crimson reds over deep blacks.
![Bloody Filter Example](media\cat_dithered_bloody.png){ width="70%" }

---

## Video & Animation Showcase

`BayerDithering` is fully optimized for hardware-accelerated video rendering. When running via the GPU backend (`-a gpu`), processing batches of video frames is blazing fast.

### Animated GIFs

When processing GIFs, transparency settings are automatically sanitized to prevent frame overlapping or smearing effects.

![Dithered GIF Demo](media\cat-plink_dithered.gif)

### Lossless Video Outputs

By default, the command line utility outputs lossless container streams to preserve high-fidelity pixel matrices without compression artifacts bleeding into the dither patterns.

<video width="80%" controls autoplay loop muted>
  <source src="../media/cat-french-web.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
