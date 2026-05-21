import argparse
import sys
import os
from BayerDithering import BayerDither, CPUProcessor, GPUProcessor, DitherConfig, matrices
from BayerDithering.utils import ProcessedVideo, ProcessedGIF, load_filters
from BayerDithering.gpu import TAICHI_AVAILABLE
import cv2
from pathlib import Path
import filetype
import imageio


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    filters = load_filters()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = "Applies a Bayer matrix dithering effect to images or videos."

    parser.add_argument('-i', '--input', metavar='PATH', required=True, help='specifies the input file (image or video) to apply the dithering effect.')
    parser.add_argument('-m', '--matrix', metavar='MATRIX', default='4x4', choices=list(matrices.keys()), help='selects the Bayer matrix size to use for dithering.')
    parser.add_argument('-a', '--arch', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('-f', '--filter', metavar='FILTER', default=None, choices=list(filters.keys()), help='applies a color filter to the output image.')
    parser.add_argument('-o', '--output', metavar='PATH', default=None, help='specifies the output file path. If not set, a default name will be used.')
    parser.add_argument('-s', '--sharpness', metavar='FACTOR', type=float, default=1.6, help="adjusts the sharpness of the image.")
    parser.add_argument('-c', '--contrast', metavar='FACTOR', type=float, default=1.5, help='adjusts the contrast of the image.')
    parser.add_argument('-d', '--downscale', metavar='FACTOR', type=int, default=2, help='downscales the image by the given factor before applying the dithering.')
    parser.add_argument('-u', '--upscale', metavar='BOOLEAN', type=str2bool, default=True, help='upscales the image back to its original size after dithering. This is useful for preserving the original dimensions of the Image.')
    parser.add_argument('-q', '--quiet', action='store_true', help='runs the script in quiet mode.')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    kind = filetype.guess(args.input)
    if kind is None:
        print("Error: Could not determine the file type of the input.")
        sys.exit(1)

    mime = kind.mime
    input_path = Path(args.input)

    if args.output is None:
        if mime == "image/gif":
            args.output = f"{input_path.stem}_dithered.gif"
        elif mime.startswith("video"):
            args.output = f"{input_path.stem}_dithered.mp4"
        elif mime.startswith("image"):
            args.output = f"{input_path.stem}_dithered.png"
        else:
            print(f"Error: Unsupported file format ({mime}).")
            sys.exit(1)
    

    filter_chosen = filters.get(args.filter) if args.filter else None
    
    config = DitherConfig(
        b_matrix=matrices[args.matrix],
        contrast=args.contrast,
        sharpness=args.sharpness,
        downscale_factor=args.downscale,
        upscale=args.upscale,
        filter=filter_chosen
    )

    processor = CPUProcessor(config)
    if args.arch == 'gpu':
        if TAICHI_AVAILABLE:
            processor = GPUProcessor(config)
        else:
            print("Warning: GPU backend (Taichi) is unavailable.\n"
                "Automatic fallback to CPU processing activated...\n"
                "Tip: Install with 'pip install BayerDithering[gpu]' to enable hardware acceleration.\n")

    ditherer = BayerDither(processor=processor, verbose=not args.quiet)

    try:
        if mime == "image/gif":
            media = imageio.get_reader(args.input)
        elif mime.startswith("video"):
            media = cv2.VideoCapture(args.input)
        elif mime.startswith("image"):
            media = cv2.imread(args.input)

        result = ditherer.apply(media)
        
        if isinstance(result, ProcessedVideo):
            media: cv2.VideoCapture
            result: ProcessedVideo
            with result:
                result.save_with_audio(original_video_path=args.input, path=args.output)
            media.release()
            
        elif isinstance(result, ProcessedGIF):
            media: imageio.Reader
            result: ProcessedGIF
            with result:
                result.save(dest_path=args.output)
            media.close()
            
        else:
            cv2.imwrite(args.output, result)

        if not args.quiet:
            print(f"Success! Processed file saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()