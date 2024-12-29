import numpy as np
import os
import multiprocessing

# Set the number of CPU cores to avoid joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
import os
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


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
    data, labels = np.array(data), np.array(labels)
    # Apply SMOTE to balance the dataset
    from imblearn.over_sampling import SMOTE
    smote = SMOTE()
    data, labels = smote.fit_resample(data.reshape(len(data), -1), labels)
    data = data.reshape(-1, 128, 128, 1)  # Reshape to original dimensions
    return data, labels


def create_cnn_model(input_shape):
    """
    Create a CNN model for analyzing frequency domain images.
    """
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
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Generate a standardized filename for model files.
def generate_model_filename(base_name="fft_cnn_model"):
    version = 1
    while os.path.exists(os.path.join(model_dir, f"{base_name}_v{version}.h5")):
        version += 1
    return f"{base_name}_v{version}.h5"


if __name__ == "__main__":
    dataset_path = "CASIA2"
    dataset_path_val = "CASIA2/dataset_split/val"

    # Load and preprocess dataset
    print("Loading dataset...")
    data, labels = load_dataset_fft(dataset_path)
    print("Dataset loaded.")

    print("Loading validation dataset...")
    val_data, val_labels = load_dataset_fft(dataset_path)
    val_data = val_data[..., np.newaxis]  # Add channel dimension
    print("Validation dataset loaded.")

    # Apply cross-validation
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    # Model file naming scheme
    base_name = "fft_cnn_model"
    model_dir = "trained_models"
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, generate_model_filename(base_name))

    # Create and train the CNN model
    model = create_cnn_model(input_shape=(128, 128, 1))
    print("Training model...")
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Augmentacja danych
    augmentor = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    augmentor.fit(data)

    for train_index, val_index in skf.split(data, labels):
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        model.fit(augmentor.flow(X_train, y_train, batch_size=32), epochs=80, validation_data=(X_val, y_val),
                  shuffle=True)
    print("Model training completed.")

    # Save the trained model
    model.save(model_file)
    print(f"Model saved as '{model_file}'")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(val_data, val_labels)
    from sklearn.metrics import accuracy_score

    predictions = model.predict(val_data)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    accuracy = accuracy_score(val_labels, predicted_classes)
    print(f"Validation Accuracy (calculated separately): {accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
