# Installation

## Prerequisites

- **Python:** Version `>= 3.13` (Support for `3.14` is currently pending `taichi` backend C-API compiled wheels, but the CPU pipeline works universally).
- **Hardware:** A dedicated GPU is highly recommended for video and GIF processing, though a CPU fallback is natively provided.

## Installation methods

### Option 1: Install via PyPI (Recommended)

The easiest way to install the library and the global CLI is directly from the Python Package Index:

```bash
pip install BayerDithering
```

To enable GPU Hardware Acceleration (Recommended for Videos), install the package with the optional `taichi` backend dependency:

```bash
pip install "BayerDithering[gpu]"
```

### Option 2: Build from Source

1. Clone the repository:

    ```bash
    git clone https://github.com/madmattp/Bayer-Dithering.git
    ```

2. Install the package and its dependencies:

    ```bash
    pip install -e .
    ```

    *(This will install the required libraries and link the `dither` command to your system).*

3. (Optional) Enable GPU Hardware Acceleration:

    ```bash
    pip install -e ".[gpu]"
    ```
