import tensorflow as tf
import numpy as np
import cv2
from collections import deque

# Load the model
model = tf.keras.models.load_model('models/cnn_lstm_drowsiness_model.h5')
print("✅ Model loaded.")

# Parameters
IMG_SIZE = 64
TIMESTEPS = 4
labels = ["Drowsy", "Not Drowsy"]

# Initialize frame buffer (sliding window of 4 frames)
frame_buffer = deque(maxlen=TIMESTEPS)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("❌ Cannot open webcam")

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Preprocess frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    frame_buffer.append(img)

    label_text = "⏳ Collecting frames..."
    color = (255, 255, 0)

    # Once we have enough frames, predict
    if len(frame_buffer) == TIMESTEPS:
        sequence = np.array(frame_buffer)  # shape: (4, 64, 64, 3)
        sequence = np.expand_dims(sequence, axis=0)  # shape: (1, 4, 64, 64, 3)

        # Predict
        preds = model.predict(sequence, verbose=0)
        pred_class = np.argmax(preds)
        confidence = np.max(preds)

        label_text = f"{labels[pred_class]} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if labels[pred_class] == "Not Drowsy" else (0, 0, 255)

    # Show prediction on frame
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Driver Drowsiness Detection', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
