import os
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Parameters
DATASET_DIR = "datasets/DriverDrowsinessDataset"
IMG_SIZE = (64, 64)  # Resize to 64x64
MAX_IMAGES_PER_CLASS = 2000  # Limit number of images per class

def load_images_from_folder(folder, label, max_images):
    images = []
    count = 0
    for filename in os.listdir(folder):
        if count >= max_images:
            break
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append((img, label))
            count += 1
    return images

def load_data():
    data = []
    
    notdrowsy_path = os.path.join(DATASET_DIR, "notdrowsy")
    drowsy_path = os.path.join(DATASET_DIR, "drowsy")

    print("[INFO] Loading 'notdrowsy' images...")
    data.extend(load_images_from_folder(notdrowsy_path, 0, MAX_IMAGES_PER_CLASS))

    print("[INFO] Loading 'drowsy' images...")
    data.extend(load_images_from_folder(drowsy_path, 1, MAX_IMAGES_PER_CLASS))

    print(f"[INFO] Total images loaded: {len(data)}")

    # Shuffle data
    random.shuffle(data)

    # Split into X (features) and y (labels)
    X = np.array([item[0] for item in data])
    y = np.array([item[1] for item in data])

    # Normalize images (important)
    X = X / 255.0

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
