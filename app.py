import os
import cv2
import numpy as np
import base64
import time
from flask import Flask, render_template, request, jsonify, session, send_file

app = Flask(__name__)
app.secret_key = 'dip_project_final_key'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- HELPER FUNCTIONS ---

def calculate_histogram(img):
    """Calculates RGB histograms for the live graph."""
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().tolist()
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten().tolist()
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten().tolist()
    return {'r': hist_r, 'g': hist_g, 'b': hist_b}

def apply_auto_levels(img):
    """Linear Contrast Stretching (Safe Auto-Enhance)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    min_val, max_val = np.percentile(l, (1, 99))
    if max_val - min_val > 0:
        l_stretched = (l.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
        l_stretched = np.clip(l_stretched, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge((l_stretched, a, b)), cv2.COLOR_LAB2BGR)
    return img

def apply_dip_algorithms(img, data):
    """Master processing function."""
    if img is None: return None

    # 1. GEOMETRY
    rotate_val = int(data.get('rotation', 0))
    if rotate_val == 90: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_val == 180: img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotate_val == 270: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    flip_val = data.get('flip', 'none')
    if flip_val == 'horizontal': img = cv2.flip(img, 1)
    elif flip_val == 'vertical': img = cv2.flip(img, 0)

    # Scaling
    scale_percent = int(data.get('scale', 100))
    interp_method = data.get('interpolation', 'linear')
    interp_map = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC}
    
    if scale_percent != 100:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height), interpolation=interp_map.get(interp_method, cv2.INTER_LINEAR))

    # 2. SEGMENTATION (Thresholding)
    threshold_mode = data.get('threshold', 'none')
    if threshold_mode != 'none':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if threshold_mode == 'binary': _, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif threshold_mode == 'otsu': _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_mode == 'adaptive': img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 3. MORPHOLOGY
    morph_op = data.get('morph', 'none')
    if morph_op != 'none':
        k_size = int(data.get('morph_size', 3))
        kernel = np.ones((k_size, k_size), np.uint8)
        if morph_op == 'erosion': img = cv2.erode(img, kernel, iterations=1)
        elif morph_op == 'dilation': img = cv2.dilate(img, kernel, iterations=1)
        elif morph_op == 'opening': img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        elif morph_op == 'closing': img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        elif morph_op == 'gradient': img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # 4. INTENSITY
    img = img.astype(np.float32)
    brightness = int(data.get('brightness', 0))
    if brightness != 0: img += brightness
    
    contrast = float(data.get('contrast', 1.0))
    if contrast != 1.0: img = (img - 127.0) * contrast + 127.0
    
    img = np.clip(img, 0, 255)

    shadow_boost = int(data.get('shadows', 0))
    highlight_cut = int(data.get('highlights', 0))
    if shadow_boost > 0 or highlight_cut > 0:
        norm = img / 255.0
        if shadow_boost > 0: norm += (shadow_boost/200.0) * (1.0 - norm)**2
        if highlight_cut > 0: norm -= (highlight_cut/200.0) * norm**2
        img = np.clip(norm * 255.0, 0, 255)

    img = img.astype(np.uint8)

    gamma = float(data.get('gamma', 1.0))
    if gamma != 1.0:
        invGamma = gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)

    # 5. FILTERS
    filter_type = data.get('filter', 'none')
    if filter_type == 'blur': img = cv2.GaussianBlur(img, (15, 15), 0)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
    elif filter_type == 'canny':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(gray, 100, 200)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'clahe':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    elif filter_type == 'levels':
        img = apply_auto_levels(img)

    return img

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', filename=session.get('filename', None))

@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    file = request.files.get('file')
    if file and file.filename != '':
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session['filename'] = filename
        return jsonify({'status': 'success', 'filename': filename})
    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/process_live', methods=['POST'])
def process_live():
    filename = session.get('filename', None)
    if not filename: return jsonify({'error': 'No Image'}), 400
    
    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    if img is None: return jsonify({'error': 'File not found'}), 404

    # Process
    processed_img = apply_dip_algorithms(img, request.json)

    # Resize for Preview Speed (Max 600px)
    h, w = processed_img.shape[:2]
    if w > 600:
        scale = 600 / w
        processed_img = cv2.resize(processed_img, (600, int(h * scale)))

    hist_data = calculate_histogram(processed_img)
    _, buffer = cv2.imencode('.jpg', processed_img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_str, 'histogram': hist_data})

@app.route('/download_image', methods=['POST'])
def download_image():
    """Saves the FULL RESOLUTION image for download"""
    filename = session.get('filename', None)
    if not filename: return "No Image", 400

    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # Process Full Resolution (No resizing down)
    final_img = apply_dip_algorithms(img, request.json)
    
    # Save to a temp path
    save_path = os.path.join(RESULT_FOLDER, f"processed_{int(time.time())}.jpg")
    cv2.imwrite(save_path, final_img)
    
    return send_file(save_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
