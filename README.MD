# Iris Detection Using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) for iris detection. The model is trained on the CASIA 1000 dataset with 20,000 images, achieving 93% accuracy.

## Project Structure

```
app
│   ├── templates
│   │   ├── uploads               # Stores uploaded images
│   │   ├── about.html            # About page
│   │   ├── index.html            # Main page for uploads
│   │   └── predict.html          # Displays prediction results
│   ├── app.py                    # Main Flask application script
│   ├── classes.npy               # Numpy array containing class labels
│   └── label_encoder             # Label encoder file

models
│   ├── encoding
│   │   └── label_encoder.joblib  # Joblib serialized label encoder
│   └── IRISRecognizer.h5         # Trained CNN model

notebooks
│   └── main.ipynb                # Jupyter notebook for model training and experimentation

.gitignore                         # Lists files to ignore in version control
readme.md                         # Project documentation
```

## Model Information

- **Model**: Convolutional Neural Network (CNN)
- **Dataset**: CASIA 1000 Iris Image Dataset (20,000 images)
- **Accuracy**: 93%
- **Model File**: `IRISRecognizer.h5` (saved model)
- **Label Encoder**: Stored in `label_encoder.joblib`

## Key Files

- **app.py**: Flask app for running the iris detection service.
- **index.html**: Frontend for image uploads.
- **predict.html**: Displays prediction results.

---
