# Image Manipulation Detection using CNN and FFT  

This project uses Convolutional Neural Networks (CNN) and Fast Fourier Transform (FFT) to detect image manipulation. The model is trained to classify images as either original or manipulated.  

## Project Overview  
- **Goal**: Develop a deep learning model capable of detecting manipulated images using frequency analysis.
- **Dataset**: The project utilizes the **CASIA2** dataset, which contains a variety of manipulated and original images.    
- **Techniques**:  
  - FFT (Fast Fourier Transform) to preprocess images.  
  - CNN (Convolutional Neural Network) for image classification.  
  - Data augmentation to improve model generalization.  
  - SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.  

## Requirements  
Ensure you have the following packages installed to run the project:  

```bash
tensorflow==2.18.0
numpy==2.0.2
opencv-python==4.10.0.84
scikit-learn==1.6.0
imbalanced-learn==0.13.0
matplotlib==3.10.0
Pillow==11.0.0
```

Alternatively, install all dependencies at once:
```
pip install -r requirements.txt
```


Run the training script:
```
python training_CNN.py
```

Evaluate the model on the validation set:
```
python evaluate_model_accuracy.py
```
