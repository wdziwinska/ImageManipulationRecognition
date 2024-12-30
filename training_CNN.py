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

    # Sprawdzenie liczby klas przed użyciem SMOTE
    unique_classes, counts = np.unique(labels, return_counts=True)
    if len(unique_classes) < 2:
        print("Warning: Not enough class diversity for SMOTE. Skipping SMOTE.")
        return data, labels  # Zwracamy dane bez SMOTE, jeśli nie można go zastosować

    # Jeśli dane są prawidłowe, stosujemy SMOTE
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
        Dense(1, activation='sigmoid')  # Binary classification
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

    # Load and preprocess dataset
    print("Loading dataset...")
    data, labels = load_dataset_fft(dataset_path)
    print("Dataset loaded.")

    print("Loading validation dataset...")
    val_data, val_labels = load_dataset_fft(dataset_path_val)
    val_data = val_data[..., np.newaxis]  # Add channel dimension
    print("Validation dataset loaded.")

    # Apply cross-validation
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Model file naming scheme
    base_name = "fft_cnn_model"
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, generate_model_filename(base_name))

    # Create and train the CNN model
    model = create_cnn_model(input_shape=(128, 128, 1))
    print("Training model...")

    # Augmentacja danych
    augmentor = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    augmentor.fit(data)

    # for train_index, val_index in skf.split(data, labels):
    #     X_train, X_val = data[train_index], data[val_index]
    #     y_train, y_val = labels[train_index], labels[val_index]
    #     model.fit(augmentor.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val), shuffle=True)
    # print("Model training completed.")
    #
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # model.fit(augmentor.flow(data, labels, batch_size=32), epochs=50, validation_data=(val_data, val_labels),
    #           callbacks=[early_stopping], shuffle=True)
    # print("Model training completed.")

    for train_index, val_index in skf.split(data, labels):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(augmentor.flow(X_train, y_train, batch_size=32), epochs=30,
                  validation_data=(X_val, y_val), callbacks=[early_stopping], shuffle=True)

    print("Model training completed.")

    # Save the trained model
    model.save(model_file)
    print(f"Model saved as '{model_file}'")

    # Evaluate the model
    # test_loss, test_accuracy = model.evaluate(X_val, y_val)
    # predictions = model.predict(X_val)
    # predicted_classes = (predictions > 0.5).astype(int).flatten()
    # accuracy = accuracy_score(y_val, predicted_classes)
    # print(f"Validation Accuracy (calculated separately): {accuracy * 100:.2f}%")
    # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
