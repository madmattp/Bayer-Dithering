from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    max_threads = os.cpu_count()  # Limite de threads baseado no número de núcleos

    if request.method == 'POST':
        # Recebe o arquivo
        file = request.files['input']

        # Recebe os demais parâmetros
        matrix = request.form['matrix']
        filter_type = request.form['filter']
        sharpness = float(request.form['sharpness'])
        contrast = float(request.form['contrast'])
        downscale = int(request.form['downscale'])
        threads = int(request.form['threads'])
        
        if file:
            pass
        
        return redirect(url_for('success'))

    return render_template('index.html', max_threads=max_threads)

@app.route('/success')
def success():
    return "TBI"

if __name__ == '__main__':
    app.run(debug=True)
