import os
import random
import shutil

# Ścieżki do folderów źródłowych
manipulated_dir = "CASIA2/fft_spectrum/manipulated"
original_dir = "CASIA2/fft_spectrum/original"

# Ścieżki do folderów docelowych
output_dir = "CASIA2/fft_spectrum/dataset_split"
test_dir = os.path.join(output_dir, "test")
val_dir = os.path.join(output_dir, "val")

manipulated_test_dir = os.path.join(test_dir, "manipulated")
manipulated_val_dir = os.path.join(val_dir, "manipulated")

original_test_dir = os.path.join(test_dir, "original")
original_val_dir = os.path.join(val_dir, "original")

# Tworzenie folderów docelowych
for dir_path in [manipulated_test_dir, manipulated_val_dir, original_test_dir, original_val_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Funkcja do podziału plików sumarycznie
def split_data_sumarycznie(manipulated_dir, original_dir, test_dir, val_dir, test_split=0.1, val_split=0.1):
    # Pobranie wszystkich plików z obu folderów
    manipulated_files = [(os.path.join(manipulated_dir, f), "manipulated") for f in os.listdir(manipulated_dir)]
    original_files = [(os.path.join(original_dir, f), "original") for f in os.listdir(original_dir)]

    all_files = manipulated_files + original_files
    random.shuffle(all_files)

    test_split_index = int(len(all_files) * test_split)
    val_split_index = test_split_index + int(len(all_files) * val_split)

    test_files = all_files[:test_split_index]
    val_files = all_files[test_split_index:val_split_index]

    # Przenoszenie plików do odpowiednich folderów
    for file_path, category in test_files:
        target_dir = manipulated_test_dir if category == "manipulated" else original_test_dir
        shutil.move(file_path, target_dir)

    for file_path, category in val_files:
        target_dir = manipulated_val_dir if category == "manipulated" else original_val_dir
        shutil.move(file_path, target_dir)

# Podział plików sumarycznie
split_data_sumarycznie(manipulated_dir, original_dir, test_dir, val_dir)

print("Podział zbioru na testowy i walidacyjny został zakończony.")
