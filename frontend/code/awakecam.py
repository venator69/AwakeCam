from flask import Flask, Response, render_template
from picamera2 import Picamera2
import cv2
import time
import subprocess
import requests
import pygame
from ultralytics import YOLO

app = Flask(__name__)

# Initialize Picamera2 for Raspberry Pi camera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(main={"size": (600, 400), "format": 'RGB888'})
picam2.configure(preview_config)
picam2.start()

# Start ngrok tunnel
def start_ngrok():
    subprocess.Popen(["ngrok", "http", "5000"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(2)

# Retrieve public URL from ngrok local API
def get_public_url(max_retries=10, delay=1):
    for _ in range(max_retries):
        try:
            res = requests.get("http://localhost:4040/api/tunnels").json()
            for tunnel in res.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    return tunnel.get("public_url")
        except Exception:
            time.sleep(delay)
    return "http://localhost:5000"

start_ngrok()
public_url = get_public_url()
print("Ngrok Public URL:", public_url)

# Initialize pygame mixer for alerts
pygame.mixer.init()
drowsy_sound = pygame.mixer.Sound("drowsiness-detected.mpeg")
sleep_sound = pygame.mixer.Sound("sleep-detected.mpeg")
alert_interval = 3  # seconds
last_drowsy = 0
last_sleep = 0

# Load detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
race_classifier = YOLO('best.pt')
class_names = ['Normal', 'Drowsy', 'Sleep']

@app.route('/')
def index():
    return render_template('testgeolive.html', public_url=public_url)

@app.route('/video_feed')
def video_feed():
    def generate():
        global last_drowsy, last_sleep
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Drowsiness detection
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            now = time.time()
            drowsy = False
            sleeping = False
            for (x, y, w, h) in faces:
                roi = frame_bgr[y:y+h, x:x+w]
                resized = cv2.resize(roi, (224,224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                res = race_classifier(rgb)
                idx = res[0].probs.top1
                conf = res[0].probs.top1conf.item()
                label = class_names[idx] if idx < len(class_names) else 'Unknown'
                cv2.rectangle(frame_bgr, (x,y), (x+w,y+h), (0,255,0),2)
                cv2.putText(frame_bgr, f"{label}:{conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
                if label == 'Drowsy': drowsy = True
                if label == 'Sleep': sleeping = True

            # Alerts
            if drowsy and now - last_drowsy >= alert_interval:
                pygame.mixer.Sound.play(drowsy_sound)
                last_drowsy = now
            if sleeping and now - last_sleep >= alert_interval:
                pygame.mixer.Sound.play(sleep_sound)
                last_sleep = now

            _, buf = cv2.imencode('.jpg', frame_bgr)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
