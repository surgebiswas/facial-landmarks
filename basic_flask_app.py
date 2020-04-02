import os
import glob

from flask import Flask, flash, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename

from flask import send_from_directory

import mask



UPLOAD_FOLDER = './images/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            input_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], 
                    secure_filename(input_file) + '_mask.jpg')
            
            # remove contents of UPLOAD_FOLDER, which will have previous data
            ufiles = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
            for uf in ufiles:
                os.remove(uf)

            file.save(input_file)
            
            # DO IMAGE PROCESSING HERE
            mask.impose_mask(input_file, output_file)
            
            return send_from_directory(app.config['UPLOAD_FOLDER'], 
                    os.path.basename(output_file), mimetype='image/jpg')
            
            
    return '''
    <!doctype html>
    <title>Maskify Me</title>
    <h1>Upload a (profile) photo of yourself, and prepare to be maskified!</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True, mimetype='image/jpg')