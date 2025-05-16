import tensorflow as tf
import numpy as np
import cv2

# Load the model
model = tf.keras.models.load_model('models/driver_drowsiness_cnn.h5')
print("✅ Model loaded.")

# Labels
labels = ["Drowsy", "Not Drowsy"]

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

    # Resize and preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img, verbose=0)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)

    label_text = f"{labels[pred_class]} ({confidence*100:.1f}%)"
    
    # Show label on the frame
    color = (0, 255, 0) if labels[pred_class] == "Not Drowsy" else (0, 0, 255)
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the frame
    cv2.imshow('Driver Drowsiness Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and close
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
