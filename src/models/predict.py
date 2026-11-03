import tensorflow as tf
import numpy as np
import cv2
from collections import deque

# --- Configuration Parameters ---
MODEL_PATH = 'models/cnn_lstm_drowsiness_model.h5'
IMG_SIZE = 64
TIMESTEPS = 4
LABELS = ["Drowsy", "Not Drowsy"]
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PREDICTION_HISTORY_LENGTH = 10 # For temporal smoothing (e.g., last 10 predictions)

# --- Load the model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure 'models/cnn_lstm_drowsiness_model.h5' exists.")
    exit()

# --- Load Face Detector ---
try:
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Could not load face cascade from {FACE_CASCADE_PATH}")
    print("Face cascade loaded.")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    print("Please ensure OpenCV's Haar Cascade XML files are accessible.")
    exit()

# --- Initialize frame buffer and prediction history ---
frame_buffer = deque(maxlen=TIMESTEPS)
prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam. Please ensure it's connected and not in use.")

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    label_text = "⏳ Collecting frames..."
    color = (255, 255, 0) # Yellow for initial state

    if len(faces) > 0:
        # Assuming only one main face for simplicity, pick the largest one
        (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])

        # Draw bounding box around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue bounding box

        # Crop and preprocess the face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Check if ROI is valid (not empty)
        if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
            img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            frame_buffer.append(img)
        else:
            label_text = "⚠️ Invalid Face ROI"
            color = (0, 165, 255) # Orange

        # Once we have enough frames, predict
        if len(frame_buffer) == TIMESTEPS:
            sequence = np.array(frame_buffer)  # shape: (4, 64, 64, 3)
            sequence = np.expand_dims(sequence, axis=0)  # shape: (1, 4, 64, 64, 3)

            # Predict
            preds = model.predict(sequence, verbose=0)
            current_pred_class = np.argmax(preds)
            current_confidence = np.max(preds)

            # Add current prediction to history for temporal smoothing
            prediction_history.append(current_pred_class)

            # Temporal Smoothing: Determine the most frequent prediction in history
            if len(prediction_history) > 0:
                # Count occurrences of each class
                class_counts = {label: 0 for label in range(len(LABELS))}
                for p in prediction_history:
                    class_counts[p] += 1
                
                # Get the most common class
                smoothed_pred_class = max(class_counts, key=class_counts.get)
                
                label_text = f"{LABELS[smoothed_pred_class]} (Current: {LABELS[current_pred_class]} {current_confidence*100:.1f}%)"
                color = (0, 255, 0) if LABELS[smoothed_pred_class] == "Not Drowsy" else (0, 0, 255) # Green for Not Drowsy, Red for Drowsy
            else:
                label_text = f"{LABELS[current_pred_class]} ({current_confidence*100:.1f}%)"
                color = (0, 255, 0) if LABELS[current_pred_class] == "Not Drowsy" else (0, 0, 255)

    else:
        label_text = "No face detected"
        color = (0, 255, 255) # Cyan when no face is detected
        # Clear frame buffer if no face is detected to prevent old frames from being used
        frame_buffer.clear()
        prediction_history.clear()


    # Show prediction on frame
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Driver Drowsiness Detection', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")