from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from dither import *
from io import BytesIO
import base64
import imageio

app = Flask(__name__)

max_threads = os.cpu_count()


def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_video_(file_extension):
    if file_extension in ['.mp4', '.avi', '.mov']:
        return True
    return False
    
@app.route('/')
def main():
    av_filters = load_filters()
    return render_template('index.html', max_threads=max_threads, matrices=matrices.keys(), filters=av_filters.keys())

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['input']
        filename = file.filename
        file_extension = get_file_extension(filename)

        selected_matrix = request.form['matrix']
        filter_type = request.form['filter']
        sharpness = float(request.form['sharpness'])
        contrast = float(request.form['contrast'])
        downscale = int(request.form['downscale'])
        threads = int(request.form['threads'])
        
        if threads > max_threads:
            threads = max_threads

        av_filters = load_filters()
        filter_chosen = av_filters[filter_type] if filter_type != 'Tux' else None
            
        matrix = matrices[selected_matrix]

        if file:
            file_stream = BytesIO(file.read())

            if is_image(file_stream):
                dithered_image = image_processing(
                    image_path=file_stream,
                    contrast=contrast,
                    sharpness=sharpness,
                    downscale_factor=downscale,
                    matrix=matrix,
                    chosen_filter=filter_chosen
                )
                output = BytesIO()
                dithered_image.save(output, format='PNG')
                output.seek(0)
                img_base64 = base64.b64encode(output.getvalue()).decode('utf-8')

                return redirect(url_for('display_image', img_data=img_base64))

            elif is_gif(file_stream):
                gif_buffer = gif_processing(
                    input_gif=file_stream,
                    contrast=contrast,
                    sharpness=sharpness,
                    downscale_factor=downscale,
                    matrix=matrix,
                    chosen_filter=filter_chosen
                )
                gif_filename = f"dithered_gif.gif"
                with open(f'static/{gif_filename}', "wb") as f:
                    f.write(gif_buffer.getvalue())

                return redirect(url_for('display_gif', gif_path=gif_filename))

            
            elif is_video_(file_extension):
                
                temp = f'temp_video{file_extension}'
                with open(f'./static/{temp}', "wb") as f:
                    f.write(file_stream.getvalue())

                video = video_processing(
                    video_path=f'./static/{temp}',
                    threads=threads,
                    contrast=contrast,
                    sharpness=sharpness,
                    downscale_factor=downscale,
                    matrix=matrix,
                    chosen_filter=filter_chosen)
            
                video_output_path = "static/dithered_video.mp4"
                video.write_videofile(video_output_path, codec='libx264')
                return redirect(url_for('display_video', video_path=video_output_path))
        
            else:
                return jsonify({"error": "Unsupported file type"}), 415

                
@app.route('/image')
def display_image():
    img_data = request.args.get('img_data')
    return render_template('display_image.html', img_data=img_data)

@app.route('/gif')
def display_gif():
    gif_path = request.args.get('gif_path')
    return render_template('display_gif.html', gif_path=gif_path)

@app.route('/video')
def display_video():
    video_path = request.args.get('video_path', None)
    return render_template('display_video.html', video_path=video_path)

def run_app():
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)
