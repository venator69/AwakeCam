import cv2
from ultralytics import YOLO
import numpy as np

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
race_classifier = YOLO('best.pt')  # Load your trained model

# Define your class mapping
#class_names = [
#    'White',    # 0
#    'China',    # 1
#    'Arab',     # 2
#    'Sunda',    # 3
#    'Jawa/Bali',# 4
#    'Indian',   # 5
#   'Batak',    # 6
#    'Papua',    # 7
#    'Black'     # 8
#]
class_names = [
    'Normal',    # 0
    'Drowsy',    # 1
    'Sleep'
]
# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with improved parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract face ROI with padding
        padding = 20
        y1 = max(0, y - padding)
        x1 = max(0, x - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x2 = min(frame.shape[1], x + w + padding)
        face_roi = frame[y1:y2, x1:x2]
        
        # Preprocess for classification
        face_resized = cv2.resize(face_roi, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Predict race
        results = race_classifier(face_rgb)
        
        # Get all probabilities
        probs = results[0].probs.data.tolist()
        
        # Get top prediction
        top1_index = results[0].probs.top1
        confidence = results[0].probs.top1conf.item()
        
        # Verify the index is within our class range
        if 0 <= top1_index < len(class_names):
            predicted_class = class_names[top1_index]
        else:
            # If invalid index, use the class with highest probability
            top1_index = probs.index(max(probs))
            predicted_class = class_names[top1_index] if top1_index < len(class_names) else "Unknown"
        
        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 255), 1)
        
        label = f"{predicted_class} ({confidence:.2f})"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow('Ethnicity Recognition', frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()