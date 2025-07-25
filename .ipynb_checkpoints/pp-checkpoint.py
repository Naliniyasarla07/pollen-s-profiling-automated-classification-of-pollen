import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder='static')
APP_ROOT = app.root_path
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = "my_model.keras"

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error="No file uploaded"), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify(error="Invalid or no selected file"), 400

    fname = secure_filename(file.filename)
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)

    try:
        img = Image.open(fpath).convert('RGB').resize((32, 32))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    except Exception as e:
        return jsonify(error=f"Failed to process image: {e}"), 400

    model = load_model(MODEL_PATH)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))

    with open("classnames.txt") as cf:
        class_names = [l.strip() for l in cf]
    pred_class = class_names[idx] if idx < len(class_names) else "Unknown"

    return jsonify(prediction=pred_class,
                   image_url=url_for('static', filename=f'uploads/{fname}'))

@app.route('/train', methods=['POST'])
def train():
    # Training logic imported from your converted notebook
    from training_module import train_cnn
    hist = train_cnn(epochs=5, batch_size=32)
    hist['model'].save(MODEL_PATH)
    return jsonify(loss=hist['loss'], accuracy=hist['accuracy'])

if __name__ == '__main__':
    app.run(debug=True)
