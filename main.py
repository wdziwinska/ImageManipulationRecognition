import cv2
import numpy as np
from sklearn.svm import SVC
from scipy.fftpack import dct
import os

class ImageManipulationDetector:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)

    def extract_features(self, image):
        """
        Extract features from the image using Discrete Cosine Transform (DCT).
        """
        if image is None:
            raise ValueError("Input image is None or cannot be read.")

        # Resize image to a fixed size for uniformity
        resized_image = cv2.resize(image, (128, 128))

        # Convert to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Apply DCT (Discrete Cosine Transform) to the image
        dct_transformed = dct(dct(gray_image.T, norm='ortho').T, norm='ortho')

        # Take the top-left corner of the DCT coefficients
        dct_features = dct_transformed[:16, :16].flatten()

        # Normalize features
        dct_features = (dct_features - np.min(dct_features)) / (np.max(dct_features) - np.min(dct_features) + 1e-7)

        return dct_features

    def train(self, dataset_path):
        """
        Train the detector using labeled CASIA v2.0 dataset.

        :param dataset_path: Path to the CASIA v2.0 dataset containing "Au" and "TP" subdirectories.
        """
        features = []
        labels = []

        for category, label in [("original", 0), ("manipulated", 1)]:
            category_path = os.path.join(dataset_path, category)
            if not os.path.exists(category_path):
                print(f"Category path does not exist: {category_path}")
                continue

            for filename in os.listdir(category_path):
                if filename.lower().endswith((".jpg", ".png", ".tif")):
                    image_path = os.path.join(category_path, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Skipping file {image_path}: Unable to read the image.")
                        continue

                    try:
                        features.append(self.extract_features(image))
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing file {image_path}: {e}")

        # Validate if features and labels are not empty
        if len(features) == 0 or len(labels) == 0:
            raise ValueError("No valid data found. Ensure the dataset path is correct and contains images.")

        # Train the model
        self.model.fit(features, labels)

    def predict(self, image_path):
        """
        Predict if the given image is manipulated or original.

        :return: 1 if manipulated, 0 if original.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read the image from {image_path}")

        features = self.extract_features(image)
        return self.model.predict([features])[0]

if __name__ == "__main__":
    detector = ImageManipulationDetector()

    # Example training setup
    dataset_path = "CASIA2"
    try:
        detector.train(dataset_path)
    except ValueError as e:
        print(f"Training failed: {e}")
        exit(1)

    # Predict example
    image_path = "CASIA2/manipulated/Tp_D_CNN_M_B_nat10139_nat00097_11948.jpg"
    try:
        prediction = detector.predict(image_path)
        print("Manipulated" if prediction == 1 else "Original")
    except ValueError as e:
        print(f"Prediction failed: {e}")
