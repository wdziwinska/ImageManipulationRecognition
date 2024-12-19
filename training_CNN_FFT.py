import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os


def preprocess_image_fft(image_path):
    """
    Preprocess an image by applying FFT and normalizing the result.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read the image: {image_path}")

    # Resize image for uniformity
    image = cv2.resize(image, (128, 128))

    # Apply FFT
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)  # Avoid log(0)

    # Normalize the result to [0, 1]
    normalized_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / \
                          (np.max(magnitude_spectrum) - np.min(magnitude_spectrum) + 1e-7)
    return normalized_spectrum


def load_dataset_fft(dataset_path):
    """
    Load dataset and preprocess images using FFT.
    """
    data = []
    labels = []
    for category, label in [("original", 0), ("manipulated", 1)]:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        for filename in os.listdir(category_path):
            if filename.lower().endswith((".jpg", ".png", ".tif")):
                image_path = os.path.join(category_path, filename)
                try:
                    spectrum = preprocess_image_fft(image_path)
                    data.append(spectrum)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")
    return np.array(data), np.array(labels)


def create_cnn_model(input_shape):
    """
    Create a CNN model for analyzing frequency domain images.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    dataset_path = "CASIA2"

    # Load and preprocess dataset
    print("Loading dataset...")
    data, labels = load_dataset_fft(dataset_path)
    data = data[..., np.newaxis]  # Add channel dimension
    print("Dataset loaded.")

    # Split into train and test sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Create and train the CNN model
    model = create_cnn_model(input_shape=(128, 128, 1))
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    print("Model training completed.")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
