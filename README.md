# EMNIST A/B Character Classifier

This project focuses on classifying the characters 'A' and 'B' from the EMNIST dataset using both traditional machine learning models and a convolutional neural network (CNN).

## Technologies Used

- Python 3
- PyTorch
- scikit-learn
- torchvision
- NumPy
- PIL (Pillow)
- Matplotlib

## Objective

The goal of this project is to perform binary classification of characters 'A' and 'B' using various methods and to compare their performance. Both classic ML models and a deep learning-based CNN model are trained and evaluated.

##  Project Workflow

- Filter only 'A' and 'B' samples from EMNIST (byclass split)
- Resize and normalize images to 28x28
- Train the following models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Convolutional Neural Network (CNN)
- Evaluate all models on the test set
- Make predictions on a custom input image

## Directory Structure

```
emnist-ab-classifier/
├── EMNIST_AB_Training_and_Evaluation.ipynb   # Model training and evaluation
├── image_prediction_with_models.ipynb        # Prediction on a custom image
├── models/
│   ├── ab_classifier_cnn.pt                  # Trained CNN model
│   ├── knn_model.pkl                         # Trained KNN model
│   ├── logistic_regression_model.pkl         # Trained Logistic Regression model
│   └── svm_model.pkl                         # Trained SVM model
├── images/
├── README.md
├── requirements.txt
```
## Results

Four different models were compared. Based on the evaluation:

- **The CNN model achieved the highest accuracy overall.**
- Among the traditional machine learning methods, **K-Nearest Neighbors (KNN)** performed the best.

All models were tested using properly preprocessed input images and successfully completed the classification task.

## Note

This project is developed for educational purposes and demonstrates binary classification using different modeling approaches.
