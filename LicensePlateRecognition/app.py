from flask import Flask, render_template, request, jsonify
import cv2
import easyocr
import sqlite3
import numpy as np
from PIL import Image
from io import BytesIO
from database import get_vehicle_info
from flask import redirect


app = Flask(__name__)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Function to create the database and table if not exists
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
    
    # Fetch details of the specific number plate
    cursor.execute("SELECT * FROM vehicle_info WHERE plate_number = ?", (numberplate,))
    data = cursor.fetchone()  # Fetch only one row
    
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
    # Connect to the database
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()
    
    # Retrieve all number plates from the database
    cursor.execute("SELECT numberplate FROM plates")
    plates = cursor.fetchall()
    
    # Close the connection
    conn.close()

    # If the form is submitted, process the number plate check
    if request.method == 'POST':
        numberplate = request.form['numberplate']
        
        # Check if the number plate is in the database
        conn = sqlite3.connect('numberplates.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plates WHERE numberplate = ?", (numberplate,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return render_template('lookup.html', plates=plates, result="Vehicle Found")
        else:
            return render_template('lookup.html', plates=plates, result="Vehicle Not Found")
    
    # If it's a GET request, just show all plates
    return render_template('lookup.html', plates=plates)


@app.route('/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return "No file uploaded", 400
    image_file = request.files['image']

    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlate_cascade = "numberplate_haarcade.xml"
    detector = cv2.CascadeClassifier(numberPlate_cascade)

    plates = detector.detectMultiScale(
        img_gray, scaleFactor=1.05, minNeighbors=7,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    img_with_boxes = img.copy()
    cropped_images = []
    alerts_found = []  # To store alerts for the frontend

    for (x, y, w, h) in plates:
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        plateROI = img[y:y+h, x:x+w]
        cropped_images.append(plateROI)

        text = reader.readtext(plateROI)
        if len(text) > 0:
            numberplate_text = text[0][1].replace(" ", "")

            # Store number plate in database
            conn = sqlite3.connect('numberplates.db')
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO plates (numberplate) VALUES (?)", (numberplate_text,))
                conn.commit()

                # Check if there's an alert for this number plate
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

            cv2.putText(img_with_boxes, numberplate_text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    _, img_with_boxes_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_with_boxes_bytes = img_with_boxes_encoded.tobytes()

    cropped_images_bytes = []
    for cropped_img in cropped_images:
        _, cropped_img_encoded = cv2.imencode('.jpg', cropped_img)
        cropped_images_bytes.append(cropped_img_encoded.tobytes())

    return jsonify({
        'full_image': img_with_boxes_bytes.decode('latin1'),
        'cropped_images': [img.decode('latin1') for img in cropped_images_bytes],
        'alerts': alerts_found  # Send alerts to the frontend
    })


@app.route('/check_numberplate', methods=['POST'])
def check_numberplate():
    numberplate = request.form.get('numberplate').replace(" ", "")

    if not numberplate:
        return "Numberplate not provided", 400

    # Check if the number plate is in the database
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

    # Store the alert in the database
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO alerts (email, numberplate) VALUES (?, ?)", (email, numberplate))
        conn.commit()
        result = "Alert added successfully. You'll be notified when this vehicle is found."
    except sqlite3.IntegrityError:
        result = "This alert already exists."

    # Fetch all plates to pass to the template
    cursor.execute("SELECT numberplate FROM plates")
    plates = cursor.fetchall()
    conn.close()

    return render_template('lookup.html', result=result, plates=plates)

@app.route('/edit_entry/<int:id>', methods=['GET', 'POST'])
def edit_entry(id):
    if request.method == 'POST':
        # Update the database with the new data
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
        # Fetch the existing data for the given ID
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
    # Connect to the database
    conn = sqlite3.connect('numberplates.db')
    cursor = conn.cursor()

    # Delete all records from the 'plates' table
    cursor.execute("DELETE FROM plates")
    cursor.execute("DELETE FROM alerts")
    cursor.execute("DELETE FROM vehicle_info")
    conn.commit()
    conn.close()

    return "Database cleared successfully!"



if __name__ == '__main__':
    app.run(debug=True)
