import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_dataset_fft(dataset_path):
    data = []
    labels = []
    for category, label in [("original", 0), ("manipulated", 1)]:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            continue
        for file in os.listdir(category_path):
            if file.endswith('.png'):
                try:
                    spectrum = cv2.imread(os.path.join(category_path, file), cv2.IMREAD_GRAYSCALE)
                    if spectrum is None:
                        print(f"Warning: Unable to read the image {file}")
                        continue
                    spectrum = cv2.resize(spectrum, (128, 128))
                    spectrum = np.expand_dims(spectrum, axis=-1)
                    data.append(spectrum)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    if len(data) == 0:
        raise ValueError(f"No valid .png files found in {dataset_path}.")

    data, labels = np.array(data), np.array(labels)

    unique_classes, counts = np.unique(labels, return_counts=True)
    if len(unique_classes) < 2:
        print("Warning: Not enough class diversity for SMOTE. Skipping SMOTE.")
        return data, labels

    smote = SMOTE()
    data, labels = smote.fit_resample(data.reshape(len(data), -1), labels)
    data = data.reshape(-1, 128, 128, 1)

    return data, labels

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_model_filename(base_name="fft_cnn_model"):
    version = 1
    while os.path.exists(os.path.join("trained_models", f"{base_name}_v{version}.h5")):
        version += 1
    return f"{base_name}_v{version}.h5"

if __name__ == "__main__":
    dataset_path = "CASIA2/fft_spectrum"
    dataset_path_val = "CASIA2/fft_spectrum/dataset_split/val"

    print("Loading dataset...")
    data, labels = load_dataset_fft(dataset_path)
    print("Dataset loaded.")

    print("Loading validation dataset...")
    val_data, val_labels = load_dataset_fft(dataset_path_val)
    val_data = val_data[..., np.newaxis]
    print("Validation dataset loaded.")

    base_name = "fft_cnn_model"
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    val_accuracies = []

    for train_index, val_index in skf.split(data, labels):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = create_cnn_model(input_shape=(128, 128, 1))

        augmentor = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        augmentor.fit(X_train)

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        history = model.fit(
            augmentor.flow(X_train, y_train, batch_size=32),
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            shuffle=True
        )

        val_accuracies.append(max(history.history['val_accuracy']))

    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.2f}")

    # Train final model on entire dataset
    model = create_cnn_model(input_shape=(128, 128, 1))
    augmentor.fit(data)
    model.fit(
        augmentor.flow(data, labels, batch_size=32),
        epochs=30,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True
    )

    model_file = os.path.join(model_dir, generate_model_filename(base_name))
    model.save(model_file)
    print(f"Model saved as '{model_file}'")
