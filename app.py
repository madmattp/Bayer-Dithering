from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        text = request.form.get('text_input')
        checkbox = request.form.get('checkbox')

        file = request.files['file_input']

        
        return redirect(url_for('success'))

    return render_template('index.html')

@app.route('/success')
def success():
    return "TBI"

if __name__ == '__main__':
    app.run(debug=True)
