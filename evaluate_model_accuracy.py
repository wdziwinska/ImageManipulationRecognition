import numpy as np
from sklearn.metrics import accuracy_score
import os
from tensorflow.keras.models import load_model
import cv2

# Ścieżka do danych
base_path = "CASIA2/fft_spectrum/dataset_split"
val_path = os.path.join(base_path, "val")
models_path = "trained_models"


# Funkcja do znalezienia wszystkich modeli
def get_all_models(models_directory):
    model_files = [f for f in os.listdir(models_directory) if f.startswith("fft_cnn_model_v") and f.endswith(".h5")]
    model_files.sort(key=lambda x: int(x.split('_v')[-1].split('.')[0]))  # Sortuj według wersji
    return [os.path.join(models_directory, f) for f in model_files]


# Funkcja wczytująca zbiór danych (bez przekształcania na FFT)
def load_dataset(dataset_path):
    """
    Load dataset without applying FFT preprocessing.
    """
    data = []
    labels = []
    for category, label in [("original", 0), ("manipulated", 1)]:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        for filename in os.listdir(category_path):
            if filename.lower().endswith((".png")):
                image_path = os.path.join(category_path, filename)
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        raise ValueError(f"Unable to read the image: {image_path}")

                    # Resize image for uniformity
                    image = cv2.resize(image, (128, 128))

                    # Normalize to [0, 1]
                    # image = image / 255.0

                    data.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")

    return np.array(data), np.array(labels)


if __name__ == "__main__":
    model_paths = get_all_models(models_path)
    if not model_paths:
        raise FileNotFoundError("Nie znaleziono żadnego modelu w katalogu: " + models_path)

    # Wczytaj i przetwórz dane walidacyjne
    print("Wczytywanie i przetwarzanie danych walidacyjnych...")
    val_data, true_classes = load_dataset(val_path)
    val_data = val_data[..., np.newaxis]

    # Iteracja po wszystkich modelach i obliczanie dokładności
    for model_path in model_paths[-6:]:
        model = load_model(model_path)
        print(f"Załadowano model: {model_path}")

        # Dokonanie predykcji na zbiorze walidacyjnym
        predictions = model.predict(val_data)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        print(predicted_classes)

        # Oblicz dokładność
        accuracy = accuracy_score(true_classes, predicted_classes)
        print(f"Dokładność dla modelu {model_path}: {accuracy * 100:.2f}%\n")
