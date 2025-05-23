import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

# Load face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Hugging Face model pipeline
pipe = pipeline("image-classification", model="chbh7051/driver-drowsiness-detection")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI with padding
        padding = 20
        y1 = max(0, y - padding)
        x1 = max(0, x - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x2 = min(frame.shape[1], x + w + padding)
        face_roi = frame[y1:y2, x1:x2]

        # Convert to PIL Image and resize as needed
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb).resize((224, 224))

        # Run classification
        results = pipe(pil_img)

        # Get top result
        if results:
            result = results[0]
            label = result['label']
            score = result['score']
        else:
            label = "Unknown"
            score = 0.0

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label} ({score:.2f})"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
