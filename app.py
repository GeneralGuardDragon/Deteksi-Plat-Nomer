from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import requests
import easyocr
from datetime import datetime
import base64

app = Flask(__name__, static_folder="static", template_folder="templates")

API_KEY = ""
MODEL_URL = ""

reader = easyocr.Reader(['en'])
history = []

def detect_plate(image_pil):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    _, buf_bgr = cv2.imencode('.jpg', img_bgr)
    bgr_b64 = base64.b64encode(buf_bgr).decode('utf-8')

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    seg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7)
    _, buf_seg = cv2.imencode('.jpg', seg)
    seg_b64 = base64.b64encode(buf_seg).decode('utf-8')

    buf = io.BytesIO()
    image_pil.save(buf, format="JPEG")
    buf.seek(0)
    response = requests.post(f"{MODEL_URL}?api_key={API_KEY}", files={"file": buf})
    if response.status_code != 200:
        return None, None, None, None

    result = response.json()
    detections = result.get("predictions", [])
    detected_texts = []
    crop_b64_list = []
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        y1 = max(0, y1)
        x1 = max(0, x1)
        y2 = min(img_bgr.shape[0], y2)
        x2 = min(img_bgr.shape[1], x2)
        plate_crop = img_bgr[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
        _, buf_crop = cv2.imencode('.jpg', plate_crop)
        crop_b64 = base64.b64encode(buf_crop).decode('utf-8')
        crop_b64_list.append(crop_b64)
        ocr_results = reader.readtext(plate_crop)
        text_detected = " ".join([res[1] for res in ocr_results]) if ocr_results else "TIDAK TERBACA"
        detected_texts.append(text_detected)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, text_detected, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    img_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".jpg", img_result)
    if not success:
        return None, None, None, None
    result_b64 = base64.b64encode(buffer).decode("utf-8")
    return result_b64, detected_texts, bgr_b64, {"segmentasi": seg_b64, "crop": crop_b64_list}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file dikirim.'}), 400
    file = request.files['file']
    try:
        image_pil = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({'error': 'Gagal membaca file gambar.'}), 400
    img_result, texts, bgr_img, stages = detect_plate(image_pil)
    if img_result is None:
        return jsonify({'error': 'Gagal memproses gambar atau mendeteksi plat nomor.'}), 500
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rec = {
        "id": int(datetime.now().timestamp() * 1000),
        "timestamp": ts,
        "texts": texts,
        "filename": file.filename
    }
    history.insert(0, rec)
    return jsonify({
        'image': img_result,
        'texts': texts,
        'timestamp': ts,
        'filename': file.filename,
        'bgr_image': bgr_img,
        'segmentasi': stages["segmentasi"],
        'crop_list': stages["crop"]
    })

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True)
