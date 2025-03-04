#!/usr/bin/python3

import numpy as np
from PIL import Image, ImageEnhance, ImageSequence
import time
from multiprocessing import Process, Manager
from moviepy.editor import VideoFileClip, ImageSequenceClip
import numpy as np
import argparse
import tomllib
import sys
import cv2
from pathlib import Path
from io import BytesIO

# PRECOMPUTED BAYER's MATRICES
bayer_matrix_2x2 = np.array([
    [0, 2],
    [3, 1]
]) / 4.0

bayer_matrix_4x4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
]) / 16.0

bayer_matrix_8x8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21]
]) / 64.0

matrices = {
    '2x2': bayer_matrix_2x2,
    '4x4': bayer_matrix_4x4,
    '8x8': bayer_matrix_8x8
}


def load_filters():
    default_filters = {
        "Orange": ((252, 176, 32), (10, 6, 3)),
        "Capuccino": ((200, 185, 150), (61, 49, 40)),
        "Brat": ((137, 205, 0), (0, 0, 0)),
        "Fairy": ((174, 255, 223), (90, 84, 117)),
        "Bloody": ((255, 42, 0), (43, 12, 0)),
        "Lavender": ((196, 167, 231), (35, 33, 54)),
        "Cyan": ((0, 204, 255), (0, 34, 43)),
        "Vapor": ((250, 185, 253), (75, 123, 222)),
        "Matrix": ((0, 255, 0), (0, 39, 6)),
        "ObraDinn": ((229, 255, 254), (51, 51, 25))
    }

    try:
        with open('filters.toml', 'rb') as file:
            filters = tomllib.load(file)
        return filters
    
    except FileNotFoundError:
        print("[FileNotFoundError] The 'filters.toml' file was not found. Using default filter pack...")
        return default_filters
    
    except tomllib.TOMLDecodeError as e:
        print(f"[TOMLDecodeError] Error parsing the TOML file: {e}\n Using default filter pack...")
        return default_filters
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.description = "Applies a Bayer matrix dithering effect to images or videos."

    parser.add_argument('-i', '--input', metavar='PATH', help='Specifies the input file (image or video) to apply the dithering effect.')
    parser.add_argument('-m', '--matrix', metavar='MATRIX', default='4x4', choices=list(matrices.keys()), help='Selects the Bayer matrix size to use for dithering. Options: 2x2, 4x4, 8x8. Default is 4x4.')
    parser.add_argument('-o', '--output', metavar='PATH', default=None, help='Specifies the output file path. If not set, a default name will be used.')
    parser.add_argument('-f', '--filter', metavar='FILTER', default=None, choices=list(load_filters().keys()), help='Applies a color filter to the output image.')
    parser.add_argument('-s', '--sharpness', metavar='FACTOR', type=float, default=1, help="Adjusts the sharpness of the image. Default is 1.")
    parser.add_argument('-c', '--contrast', metavar='FACTOR', type=float, default=1, help='Adjusts the contrast of the image. Default is 1.')
    parser.add_argument('-d', '--downscale', metavar='FACTOR', type=int, default=1, help='Downscales the image by the given factor before applying the dithering. Default is 1.')
    parser.add_argument('-t', '--threads', metavar='INTEGER', type=int, default=1, help='Specifies the number of threads to use for parallel processing. Default is 1.')

    args = parser.parse_args()
    return args


def sharpen(image: Image, factor: float):
    enhancer = ImageEnhance.Sharpness(image)
    sharp_image = enhancer.enhance(factor)
    return sharp_image

def downscale(image: Image, pot: int):
    width = image.width // pot
    height = image.height // pot
    downscaled_image = image.resize((width, height), Image.Resampling.NEAREST)
    return downscaled_image

def bayer_dithering(image: Image, bayer_matrix: np.ndarray):
    grayscale_image = image.convert('L')

    img_array = np.array(grayscale_image, dtype=np.float32) / 255.0
    height, width = img_array.shape

    for y in range(height):
        for x in range(width):
            threshold = bayer_matrix[y % len(bayer_matrix), x % len(bayer_matrix)]
        
            if round(img_array[y, x], 1) > threshold:
                img_array[y, x] = 1
            else:
                img_array[y, x] = 0
            
    dithered_image = Image.fromarray((img_array * 255).astype(np.uint8))
    return dithered_image

def colored_filter(image: Image, colors: list = None):
    image = image.convert("RGB")
    if colors is None:
        return image
    
    light = tuple(colors[0])
    dark = tuple(colors[1])

    width, height = image.size

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if pixel == (255, 255, 255):
                image.putpixel((x, y), light)
            else:
                image.putpixel((x, y), dark)

    return image

def is_image(file_path: Path):
    try:
        with Image.open(file_path) as img:
            img.verify()
            if img.format == 'GIF':
                return False
        return True
    
    except (IOError, SyntaxError):
        return False

def image_processing(image_path: Path, contrast: float, sharpness: float, downscale_factor: int, matrix: np.ndarray, chosen_filter: list = None):
    with Image.open(image_path) as image:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor=contrast)
        sharp_image = sharpen(image=image, factor=sharpness)
        downscaled_image = downscale(image=sharp_image, pot=downscale_factor)
        dithered_image = bayer_dithering(image=downscaled_image, bayer_matrix=matrix)

        if chosen_filter is not None:
            dithered_image = colored_filter(image=dithered_image, colors=chosen_filter)

        return dithered_image

def is_gif(file_path: Path):
    try:
        with Image.open(file_path) as img:
            if img.format == 'GIF':
                return True
        return False
    except (IOError, SyntaxError):
        return False
    
def gif_processing(input_gif: Path, contrast: float, sharpness: float, downscale_factor: int, matrix: np.ndarray, chosen_filter: list = None):
    with Image.open(input_gif) as gif:
        durations = gif.info['duration']
        frames = []

        for frame in ImageSequence.Iterator(gif):
            frame = frame.convert("RGB")
            enhancer = ImageEnhance.Contrast(frame)
            frame = enhancer.enhance(contrast)

            sharp_image = sharpen(image=frame, factor=sharpness)
            downscaled_image = downscale(image=sharp_image, pot=downscale_factor)
            dithered_image = bayer_dithering(image=downscaled_image, bayer_matrix=matrix)

            if chosen_filter:
                dithered_image = colored_filter(dithered_image, chosen_filter)
            
            frames.append(dithered_image)

    gif_buffer = BytesIO()
    frames[0].save(gif_buffer, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=durations, transparency=1)
    gif_buffer.seek(0)
    
    return gif_buffer

def is_video(file_path: Path):
    try:
        video = cv2.VideoCapture(file_path)
        if video.isOpened():
            return True
        return False
    except Exception:
        return False
    finally:
        video.release()

def video_processing(video_path: Path, threads: int, contrast: float, sharpness:float, downscale_factor: int, matrix: np.ndarray, chosen_filter: list = None):
    def process_frame(frame_array, contrast, sharpness, downscale_pot, bayer_matrix, chosen_filter):
        image = Image.fromarray(frame_array)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        sharp_image = sharpen(image=image, factor=sharpness)
        downscaled_image = downscale(image=sharp_image, pot=downscale_pot)
        dithered_image = bayer_dithering(image=downscaled_image, bayer_matrix=bayer_matrix)
        dithered_image = colored_filter(dithered_image, chosen_filter)

        return np.array(dithered_image)

    def process_clip(index, all_processed_frames, frames, contrast, sharpness, downscale_pot, chosen_filter, bayer_matrix):
        processed_frames = []
        for frame in frames:
            processed_frame = process_frame(frame, contrast, sharpness, downscale_pot, bayer_matrix, chosen_filter)
            processed_frames.append(processed_frame)

        all_processed_frames[index] = processed_frames

    video = VideoFileClip(video_path)
    audio_clip = video.audio
    fps = video.fps

    num_cpus = threads
    duration_per_process = video.duration / num_cpus
    manager = Manager()
    all_processed_frames = manager.dict()
    procs = []

    for i in range(num_cpus):
        start = i * duration_per_process
        end = min((i + 1) * duration_per_process, video.duration)
        
        subclip = video.subclip(start, end)
        frames = [frame for frame in subclip.iter_frames()]

        proc = Process(target=process_clip, args=(i, all_processed_frames, frames, contrast, sharpness, downscale_factor, chosen_filter, matrix))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    all_processed_frames = dict(sorted(all_processed_frames.items()))
    all_processed_frames = [frame for sublist in all_processed_frames.values() for frame in sublist]
    final_clip = ImageSequenceClip(all_processed_frames, fps=fps)
    final_clip = final_clip.set_audio(audio_clip)
    
    return final_clip

if __name__ == "__main__":
    start_time = time.time()

    args = parse_arguments()

    filters = load_filters()
    filter_chosen = filters[args.filter] if args.filter is not None else None

    try:
        bayer_matrix = matrices[args.matrix]
        if is_image(file_path=args.input):
            dithered_image = image_processing(image_path=args.input,
                            contrast=args.contrast,
                            sharpness=args.sharpness,
                            downscale_factor=args.downscale,
                            matrix=bayer_matrix,
                            chosen_filter=filter_chosen)
            output_file = args.output if args.output is not None else "dithered_image.png"
            dithered_image.save(output_file)

        elif is_gif(file_path=args.input):
            gif_buffer = gif_processing(input_gif=args.input,
                            contrast=args.contrast,
                            sharpness=args.sharpness,
                            downscale_factor=args.downscale,
                            matrix=bayer_matrix,
                            chosen_filter=filter_chosen)
            with open("dithered_gif.gif", "wb") as f:
                f.write(gif_buffer.getvalue())

        elif is_video(file_path=args.input):
            final_clip = video_processing(video_path=args.input,
                            threads=args.threads,
                            contrast=args.contrast,
                            sharpness=args.sharpness,
                            downscale_factor=args.downscale,
                            matrix=bayer_matrix,
                            chosen_filter=filter_chosen)
            output_file = args.output if args.output is not None else "dithered_video.mp4"
            final_clip.write_videofile(output_file, codec="libx264")

        else:
            raise ValueError

    except ValueError as e:
        print(f"[ ValueError ] Input file does not have a valid format!")
        sys.exit(1)

    except KeyError:
        valid_keys = ', '.join(matrices.keys())
        print(f"[ KeyError ] The matrix '{args.matrix}' was not found.")
        print(f"Valid options are: {valid_keys}")
        sys.exit(1)

    except FileNotFoundError:
        print(f"[ FileNotFoundError ] Input file {args.input} not found!")
        sys.exit(1)

    end_time = time.time() 
    execution_time = end_time - start_time  
    print(f"Processing time: {execution_time:.4f} seconds")
