from flask import Flask, render_template, request, jsonify
import cv2
import sqlite3
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
from database import get_vehicle_info
from flask import redirect
import base64
import torch
import re
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# Load YOLO model for license plate detection
yolo_model = YOLO("license_plate_detector.pt")

# Load TrOCR model for text extraction
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", use_fast=False)
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Function to create the database and table if not exists
def create_database():
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            numberplate TEXT UNIQUE NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            numberplate TEXT NOT NULL,
            UNIQUE(email, numberplate)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            owner TEXT NOT NULL,
            fines TEXT,
            accidents TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Call create_database once to set up the database
create_database()

@app.route('/details/<numberplate>')
def details(numberplate):
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_info WHERE plate_number = ?", (numberplate,))
    data = cursor.fetchone()
    conn.close()
    if data:
        return render_template('details.html', data=data)
    else:
        return render_template('details.html', data=None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/lookup', methods=['GET', 'POST'])
def lookup():
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    cursor.execute("SELECT numberplate FROM plates")
    plates = cursor.fetchall()
    conn.close()

    if request.method == 'POST':
        numberplate = request.form['numberplate']
        conn = sqlite3.connect('numberplates.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plates WHERE numberplate = ?", (numberplate,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return render_template('lookup.html', plates=plates, result="Vehicle Found")
        else:
            return render_template('lookup.html', plates=plates, result="Vehicle Not Found")
    return render_template('lookup.html', plates=plates)

@app.route('/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return "No file uploaded", 400
    image_file = request.files['image']

    # Read and decode the image
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_with_boxes = img.copy()
    cropped_images = []
    alerts_found = []

    # Run YOLO inference to detect license plates
    results = yolo_model(img, device="mps")  # Use GPU if available

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            # Crop the license plate region
            plate_crop = img[y1:y2, x1:x2]

            # Convert to PIL format for OCR
            pil_plate = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))

            # Preprocess the image for OCR
            pil_plate = pil_plate.convert("L")  # Convert to grayscale
            enhancer = ImageEnhance.Contrast(pil_plate)
            pil_plate = enhancer.enhance(2.0)  # Increase contrast
            pil_plate = pil_plate.resize((96, 24))  # Resize for TrOCR
            pil_plate = pil_plate.convert("RGB")

            # Perform OCR with TrOCR
            with torch.no_grad():
                pixel_values = trocr_processor(pil_plate, return_tensors="pt").pixel_values
                generated_ids = trocr_model.generate(pixel_values)
                numberplate_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Filter only alphanumeric characters
            numberplate_text = re.sub(r'[^A-Za-z0-9]', '', numberplate_text)

            # Store number plate in database
            conn = sqlite3.connect('numberplates.db')
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO plates (numberplate) VALUES (?)", (numberplate_text,))
                conn.commit()

                # Check for alerts
                cursor.execute("SELECT email FROM alerts WHERE numberplate = ?", (numberplate_text,))
                alerts = cursor.fetchall()
                for alert in alerts:
                    email = alert[0]
                    alerts_found.append({'email': email, 'numberplate': numberplate_text})

            except sqlite3.IntegrityError:
                pass  # Ignore duplicate entries
            conn.close()

            # Draw bounding box and text on the image
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, numberplate_text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

            # Add cropped image to the list
            cropped_images.append(plate_crop)

    # Encode images as base64
    _, img_with_boxes_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_with_boxes_base64 = base64.b64encode(img_with_boxes_encoded).decode('utf-8')

    cropped_images_base64 = []
    for cropped_img in cropped_images:
        _, cropped_img_encoded = cv2.imencode('.jpg', cropped_img)
        cropped_images_base64.append(base64.b64encode(cropped_img_encoded).decode('utf-8'))

    return jsonify({
        'full_image': img_with_boxes_base64,
        'cropped_images': cropped_images_base64,
        'alerts': alerts_found
    })

# Rest of the routes remain unchanged
@app.route('/check_numberplate', methods=['POST'])
def check_numberplate():
    numberplate = request.form.get('numberplate').replace(" ", "")
    if not numberplate:
        return "Numberplate not provided", 400
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plates WHERE numberplate = ?", (numberplate,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return "Vehicle Found"
    else:
        return "Vehicle Not Found"

@app.route('/add_alert', methods=['POST'])
def add_alert():
    numberplate = request.form['numberplate'].replace(" ", "")
    email = request.form['email']
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO alerts (email, numberplate) VALUES (?, ?)", (email, numberplate))
        conn.commit()
        result = "Alert added successfully. You'll be notified when this vehicle is found."
    except sqlite3.IntegrityError:
        result = "This alert already exists."
    cursor.execute("SELECT numberplate FROM plates")
    plates = cursor.fetchall()
    conn.close()
    return render_template('lookup.html', result=result, plates=plates)

@app.route('/edit_entry/<int:id>', methods=['GET', 'POST'])
def edit_entry(id):
    if request.method == 'POST':
        plate_number = request.form['plate_number']
        owner = request.form['owner']
        fines = request.form['fines']
        accidents = request.form['accidents']
        conn = sqlite3.connect('numberplates.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE vehicle_info
            SET plate_number = ?, owner = ?, fines = ?, accidents = ?
            WHERE id = ?
        ''', (plate_number, owner, fines, accidents, id))
        conn.commit()
        conn.close()
        return redirect(f'/details/{plate_number}')
    else:
        conn = sqlite3.connect('numberplates.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vehicle_info WHERE id = ?", (id,))
        row = cursor.fetchone()
        conn.close()
        return render_template('edit_entry.html', row=row)

@app.route('/add_entry', methods=['GET', 'POST'])
def add_entry():
    if request.method == 'POST':
        plate_number = request.form['plate_number']
        owner = request.form['owner']
        fines = request.form['fines']
        accidents = request.form['accidents']
        try:
            with sqlite3.connect('numberplates.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO vehicle_info (plate_number, owner, fines, accidents)
                    VALUES (?, ?, ?, ?)
                ''', (plate_number, owner, fines, accidents))
                conn.commit()
            return redirect(f'/details/{plate_number}')
        except sqlite3.DatabaseError as e:
            return f"Error: {e}", 500
    else:
        return render_template('add_entry.html')

@app.route('/clear_db', methods=['POST'])
def clear_db():
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM plates")
    cursor.execute("DELETE FROM alerts")
    cursor.execute("DELETE FROM vehicle_info")
    conn.commit()
    conn.close()
    return "Database cleared successfully!"

@app.route('/extract_video', methods=['POST'])
def extract_video():
    if 'video' not in request.files:
        return "No file uploaded", 400
    video_file = request.files['video']

    # Save the video file temporarily
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    detected_plates = []
    alerts_found = []  # To store alerts for the frontend

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame (e.g., every 10th frame)
        if frame_count % 10 == 0:
            # Run YOLO inference to detect license plates
            results = yolo_model(frame, device="mps")  # Use GPU if available

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the license plate region
                    plate_crop = frame[y1:y2, x1:x2]

                    # Convert to PIL format for OCR
                    pil_plate = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))

                    # Preprocess the image for OCR
                    pil_plate = pil_plate.convert("L")  # Convert to grayscale
                    enhancer = ImageEnhance.Contrast(pil_plate)
                    pil_plate = enhancer.enhance(2.0)  # Increase contrast
                    pil_plate = pil_plate.resize((96, 24))  # Resize for TrOCR
                    pil_plate = pil_plate.convert("RGB")

                    # Perform OCR with TrOCR
                    with torch.no_grad():
                        pixel_values = trocr_processor(pil_plate, return_tensors="pt").pixel_values
                        generated_ids = trocr_model.generate(pixel_values)
                        numberplate_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # Filter only alphanumeric characters
                    numberplate_text = re.sub(r'[^A-Za-z0-9]', '', numberplate_text)

                    # Store number plate in database
                    conn = sqlite3.connect('numberplates.db')
                    cursor = conn.cursor()
                    try:
                        cursor.execute("INSERT INTO plates (numberplate) VALUES (?)", (numberplate_text,))
                        conn.commit()

                        # Check for alerts
                        cursor.execute("SELECT email FROM alerts WHERE numberplate = ?", (numberplate_text,))
                        alerts = cursor.fetchall()
                        for alert in alerts:
                            email = alert[0]
                            alerts_found.append({
                                'email': email,
                                'numberplate': numberplate_text
                            })

                    except sqlite3.IntegrityError:
                        pass  # Ignore duplicate entries
                    conn.close()

                    # Store detected plate information
                    detected_plates.append({
                        'frame_number': frame_count,
                        'numberplate': numberplate_text,
                        'plate_image': plate_crop
                    })

        frame_count += 1

    cap.release()

    # Encode detected plates as base64
    detected_plates_base64 = []
    for plate in detected_plates:
        _, plate_img_encoded = cv2.imencode('.jpg', plate['plate_image'])
        plate_img_base64 = base64.b64encode(plate_img_encoded).decode('utf-8')
        detected_plates_base64.append({
            'frame_number': plate['frame_number'],
            'numberplate': plate['numberplate'],
            'plate_image': plate_img_base64
        })

    return jsonify({
        'detected_plates': detected_plates_base64,
        'alerts': alerts_found  # Send alerts to the frontend
    })

if __name__ == '__main__':
    app.run(debug=True)
