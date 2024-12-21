import numpy as np
from sklearn.metrics import accuracy_score
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ścieżka do danych
base_path = "CASIA2/dataset_split"
val_path = os.path.join(base_path, "val")
models_path = "trained_models"  # Podaj ścieżkę do katalogu z modelami

# Funkcja do znalezienia najnowszego modelu
def get_latest_model(models_directory):
    model_files = [f for f in os.listdir(models_directory) if f.startswith("fft_cnn_model_v") and f.endswith(".h5")]
    model_files.sort(key=lambda x: int(x.split('_v')[-1].split('.')[0]))  # Sortuj według wersji
    return os.path.join(models_directory, model_files[-1]) if model_files else None

# Znajdź najnowszy model
latest_model_path = get_latest_model(models_path)
if not latest_model_path:
    raise FileNotFoundError("Nie znaleziono żadnego modelu w katalogu: " + models_path)

# Przygotowanie danych
image_size = (128, 128)  # Rozmiar obrazu
batch_size = 32

# Funkcja konwertująca RGB na skalę szarości
def rgb_to_grayscale(image):
    return np.mean(image, axis=-1, keepdims=True)  # Konwersja RGB na skalę szarości

# Generator danych
val_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=rgb_to_grayscale  # Dodaj konwersję do skali szarości
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

# Mapa klas - ręczne przypisanie
class_mapping = {
    'manipulated': 1,  # Manipulowane obrazy jako klasa 1
    'original': 0      # Oryginalne obrazy jako klasa 0
}

if __name__ == "__main__":
    # Załaduj wytrenowany model
    model = load_model(latest_model_path)
    print(f"Załadowano model: {latest_model_path}")


    # Dokonanie predykcji na zbiorze walidacyjnym
    predictions = model.predict(val_generator)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    # true_classes = val_generator.classes
    true_classes = np.array([class_mapping[os.path.dirname(f).split(os.path.sep)[-1]] for f in val_generator.filenames])

    # Oblicz dokładność
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"Dokładność (sklearn) na zbiorze walidacyjnym: {accuracy * 100:.2f}%")
