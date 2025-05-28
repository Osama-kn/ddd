import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Paths
DATA_DIR = 'data/DDD'
CATEGORIES = ['Drowsy', 'NotDrowsy']
IMG_SIZE = 100

data = []
labels = []

for category in CATEGORIES:
    folder_path = os.path.join(DATA_DIR, category)
    for img_name in tqdm(os.listdir(folder_path), desc=f"Loading {category}"):
        try:
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(category)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# Convert to NumPy arrays
X = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = LabelEncoder().fit_transform(labels)

# Save preprocessed data
os.makedirs('data/processed', exist_ok=True)
with open('data/processed/processed_data.pkl', 'wb') as f:
    pickle.dump((X, y), f)

print(f"✅ Preprocessing complete. Saved {len(X)} samples to data/processed/processed_data.pkl")
