<p align="center">
  <img src="https://github.com/user-attachments/assets/54ce09b2-ee4b-4136-b265-445acea2d124" alt="Dithered Image" width="100%">
</p>


# Bayer Dithering
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)

This Python script applies a Bayer matrix dithering to images, gifs or videos. It offers a variety of customizable options such as matrix size, filters, sharpness, contrast, and downscaling. It also supports multi-threading for faster processing and includes a web interface for ease of use.

## Features
- Dithering for images and videos: Applies Bayer matrix dithering to PNG, JPG, GIF, and MP4 files.
- Customizable filters: Configurable through the filters.toml file.
- Web interface: Simple web interface for quick usage.
- Multi-threading Support: Process large files faster with parallel processing.
- Support for downscaling and sharpening: Image manipulation options before dithering.

## Installation
1. Clone the repository:
```
git clone https://github.com/madmattp/Bayer-Dithering.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage:
### Command Line Options:
- ```-i, --input```: Specifies the input file (image or video) to apply dithering to.
- ```-m, --matrix```: Selects the Bayer matrix size. Options: 2x2, 4x4, 8x8 (default: 4x4).
- ```-o, --output```: Specifies the output file path. A default name will be used if not provided.
- ```-f, --filter```: Applies a color filter to the output image.
- ```-s, --sharpness```: Adjusts the sharpness (default: 1).
- ```-c, --contrast```: Adjusts the contrast (default: 1).
- ```-d, --downscale```: Downscales the image by a factor before dithering (default: 1).
- ```-t, --threads```: Specifies the number of threads for processing (default: 1).
- ```-w, --webui```: Launches a web interface on port 5000.

### Recommended Settings
For more visually pleasing results before applying dithering, it is recommended to use the following settings:

- **Contrast**: **1.5**
- **Sharpness**: **1.6**
- **Downscaling**: **>2**

### Examples:
1. Dithering an Image with Recommended Settings:
```
python dither.py -i input_image.png -o output_image.png -m 4x4 -c 1.5 -s 1.6 -d 2
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/2f4092bf-98b6-44d5-91af-441000b3979f" alt="Silly Cat" width="45%">
  <img src="https://github.com/user-attachments/assets/c521928f-865f-4c70-9953-26b122691cea" alt="Dithered Image" width="45%">
</p>

2. Dithering a Video with Multiple Threads:
```
python dither.py -i input_video.mp4 -o output_video.mp4 -m 4x4 -t 4 -c 1.5 -s 1.6 -d 2
```
<p align="center">
  <video src="https://github.com/user-attachments/assets/1d1abe14-625d-4bd7-87e6-00cb6b14da05" width="45%" controls></video>
  <video src="https://github.com/user-attachments/assets/de229bbc-6cb2-43fb-a9d8-5c4cda07f06e" width="45%" controls></video>
</p>


3. Using the Web Interface:
```
python dither.py -w
```
or
```
python dither_web.py
```
Then, open http://localhost:5000 in your browser.

## Configuration
Fine-tune the dithering effect by adjusting matrix size, sharpness, contrast, and downscaling factors. Use multi-threading for faster video processing.

## Contributions
Feel free to open issues or contribute via pull requests. Contributions are welcome!
