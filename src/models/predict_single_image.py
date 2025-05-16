import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model('models/driver_drowsiness_cnn.h5')

# Load an image
image_path = "data/001_noglasses_nonsleepyCombination_2655_notdrowsy.jpg"  # 👈 put here a path to a real test image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (64, 64))  # 👈 Make sure it's the same size as during training

# Preprocess
image = image / 255.0
image = np.expand_dims(image, axis=0)  # shape (1, 64, 64, 3)

# Predict
predictions = model.predict(image)
predicted_class = np.argmax(predictions)

labels = ["Drowsy", "Not Drowsy"]
print(f"Prediction: {labels[predicted_class]}")
