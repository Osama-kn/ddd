import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
import os

# Load preprocessed data
with open('data/processed/processed_data.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Encode labels
y = to_categorical(y, num_classes=2)

# Create sequences (e.g. 10 frames per sample)
SEQ_LENGTH = 10
if len(X) < SEQ_LENGTH:
    raise ValueError("Not enough data to form a sequence")

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X, y, SEQ_LENGTH)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=np.argmax(y_seq, axis=1))

# Model
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(SEQ_LENGTH, X.shape[1], X.shape[2], 1)),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save
os.makedirs('models', exist_ok=True)
model.save('models/driver_drowsiness_cnn_lstm.h5')
print("✅ Model trained and saved at models/driver_drowsiness_cnn_lstm.h5")
