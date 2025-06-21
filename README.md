# Speech Emotion Classification

This project is an end-to-end pipeline for classifying emotions from speech audio data using deep learning. The system processes raw audio files, extracts comprehensive features, and uses a trained Artificial Neural Network (ANN) to predict one of eight emotional states: neutral, calm, happy, sad, angry, fearful, disgust, and surprise.

The project includes a Jupyter notebook with the full development process, a trained model, an inference script for single or batch predictions, and a Streamlit web application for interactive use.

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Feature Extraction](#2-feature-extraction)
  - [3. Feature Selection](#3-feature-selection)
  - [4. Data Augmentation and Balancing](#4-data-augmentation-and-balancing)
  - [5. Model Pipeline and Architecture](#5-model-pipeline-and-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Running the Inference Script](#running-the-inference-script)
  - [Running the Streamlit Web App](#running-the-streamlit-web-app)

## Project Overview
The primary objective is to design and implement a robust pipeline that accurately identifies and categorizes emotional states from speech. The model is trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which contains recordings of 24 professional actors vocalizing emotions.

## Methodology

### 1. Data Preparation
- **Dataset**: RAVDESS dataset. Audio filenames follow a structured naming convention that provides metadata such as emotion, actor, and gender.
- **Parsing**: A dataframe was created by parsing the filenames to extract and label the emotion for each audio file.
- **Emotion Mapping**: The numeric emotion codes from the filenames were mapped to human-readable labels (e.g., '01' -> 'neutral').

### 2. Feature Extraction
A comprehensive set of 392 features was extracted from each audio file using the `librosa` library. This ensures that a wide range of acoustic properties is captured. The features include:
- **MFCCs (40 coefficients)**: Mean and standard deviation.
- **Delta MFCCs**: Mean and standard deviation.
- **Delta-Delta MFCCs**: Mean and standard deviation.
- **Chroma Features**: Mean and standard deviation.
- **Mel Spectrogram**: Mean and standard deviation.
- **Spectral Contrast**: Mean and standard deviation.
- **Tonnetz (Tonal Centroid Features)**: Mean and standard deviation.
- **Zero Crossing Rate (ZCR)**: Mean and standard deviation.
- **Root Mean Square (RMS) Energy**: Mean and standard deviation.

### 3. Feature Selection
To reduce dimensionality and improve model performance, feature selection was performed:
- A `RandomForestClassifier` was trained on the full feature set.
- The **top 275 most important features** were selected based on the classifier's `feature_importances_`. This step helps in retaining the most discriminative features while discarding noise.

### 4. Data Augmentation and Balancing
The RAVDESS dataset is imbalanced, particularly for classes like 'disgust'. To address this, the training data was balanced using **oversampling**. After splitting the data into training and validation sets, the minority classes in the training set were oversampled to match the number of samples in the majority class. This ensures the model does not become biased towards dominant classes.

### 5. Model Pipeline and Architecture
The pipeline consists of the following steps:
1.  **Scaling**: The selected features were scaled using `StandardScaler` to normalize the data.
2.  **ANN Model**: A Sequential Artificial Neural Network (ANN) was built using TensorFlow/Keras.

The model architecture is as follows:
- **Input Layer**: `Dense(units=275, activation='relu')`
- **Hidden Layer 1**: `Dense(units=200, activation='relu')`, followed by `BatchNormalization` and `Dropout(0.3)`
- **Hidden Layer 2**: `Dense(units=150, activation='relu')`, followed by `BatchNormalization` and `Dropout(0.3)`
- **Hidden Layer 3**: `Dense(units=100, activation='relu')`, followed by `BatchNormalization` and `Dropout(0.2)`
- **Output Layer**: `Dense(units=8, activation='softmax')` for multi-class classification.

The model was compiled with the `adam` optimizer and `categorical_crossentropy` loss function. Early stopping and learning rate reduction callbacks were used during training to prevent overfitting.

## Evaluation Metrics
The model was evaluated on the validation set, achieving the following performance, which meets the project's success criteria:
- **Overall Accuracy**: **87.2%**
- **Macro F1 Score**: **0.87**
- **Per-Class F1 Scores**: All classes achieved an F1 score above 80%, indicating balanced performance across all emotions.

A confusion matrix was also generated to visualize the model's performance on a per-class basis.

## Project Structure
```
emotion_classification_project/
│
├── emotion_classification.ipynb       # Jupyter notebook with full code and explanations
├── predict_emotion.py                 # Script for single or batch inference
├── app.py                             # Streamlit web application
│
├── emotion_ann_model.h5               # Trained Keras model
├── label_encoder.pkl                  # Saved LabelEncoder object
├── scaler.pkl                         # Saved StandardScaler object
├── top_indices.pkl                    # Indices of selected features
│
├── requirements.txt                   # Required Python packages
└── README.md                          # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <(https://github.com/VK180/Speech-Emotion-Classification)>
    cd emotion_classification_project
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Inference Script
The `predict_emotion.py` script can predict emotions from a single audio file or an entire folder of audio files and outputs the accuracy.

**For a single `.wav` file:**
```bash
python predict_emotion.py "path/to/your/audio.wav"
```

**For a folder containing multiple `.wav` files:**
```bash
python predict_emotion.py "path/to/your/folder"
```



### Running the Streamlit Web App
The web app provides an interactive interface to test the model.

1.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

2.  **Use the app:**
    - Open the local URL provided by Streamlit in your browser.
    - Click "Browse files" to upload a `.wav` file.
    - The app will display the predicted emotion for the uploaded audio. 
