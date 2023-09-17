from flask import Flask, render_template, url_for, flash, request, redirect
import urllib.request
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "i-wanna-go-home"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(file.filename)
    print(file)
    
    # image = Image.open('static/uploads/'+file.filename)
    # n_i = image.resize((50, 50))
    # file = FileStorage(n_i)   
    print(file)
    if file.filename == '':
        flash('No picture selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Picture successfully uploaded')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed picture types are: png, jpg, jpeg')
        return redirect(request.url)
    
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)


