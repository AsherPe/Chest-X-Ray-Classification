# Chest X-Ray Classification Using Deep Learning

Multi-class chest X-ray classification project comparing multiple deep learning approaches for medical image analysis, including a custom CNN, transfer learning with EfficientNetB0, and a patch-based RNN architecture.

## Overview

This project focuses on classifying chest X-ray images into three categories:

- Normal
- Bacterial Pneumonia
- Viral Pneumonia

The goal was to compare different deep learning architectures and analyze how architectural choices, transfer learning strategies, and spatial representation impact medical image classification performance.

The project was developed as part of the **Fundamentals of Deep Learning** course at **The College of Management Academic Studies**.

---

# Project Goals

- Compare CNN-based and sequence-based approaches for chest X-ray classification
- Evaluate transfer learning vs training from scratch
- Analyze the effect of fine-tuning depth in EfficientNet
- Investigate how preserving spatial structure affects performance
- Handle class imbalance using augmentation and class weighting

---

# Dataset

The dataset is based on a publicly available Kaggle chest X-ray dataset.

Original classes:
- Normal
- Pneumonia

The Pneumonia class was further divided into:
- Bacterial Pneumonia
- Viral Pneumonia

using filename-based labeling.

## Final Dataset Distribution

| Split | Images |
|---|---|
| Train | 4185 |
| Validation | 1047 |
| Test | 624 |

Stratified sampling was used to preserve class balance across all subsets.

---

# Preprocessing Pipeline

All images were resized to:

```python
224 × 224
```

### CNN / RNN preprocessing
- Normalization to `[0,1]`

### EfficientNet preprocessing
- `preprocess_input()` from TensorFlow/Keras

### Data Augmentation
Applied only to the training set:
- Small rotations
- Width/height shifts
- Zoom augmentation

Horizontal flipping was intentionally avoided in order to preserve anatomical consistency in chest X-ray images.

---

# Models

# 1. Custom CNN

A custom convolutional neural network was designed from scratch.

## Architecture

The model contains:
- 4 convolutional blocks
- Batch Normalization
- Max Pooling
- Fully Connected Layers
- Dropout
- L2 Regularization
- Softmax output layer

The CNN progressively learns:
- Low-level edges
- Lung textures
- Spatial medical patterns

## Training Configuration

```python
Optimizer: Adam
Learning Rate: 1e-4
Loss: Categorical Crossentropy + Label Smoothing
Dropout: 0.5
```

### Additional Techniques
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- Class Weights

## Results

The custom CNN achieved the best performance:

```python
Test Accuracy ≈ 83.33%
```

Interestingly, removing an additional 256-filter convolutional block improved generalization and reduced overfitting, increasing accuracy from approximately 82.05% to 83.33%.

---

# 2. Transfer Learning – EfficientNetB0

EfficientNetB0 pretrained on ImageNet was used as a transfer learning backbone.

## Approach

The model used:
- EfficientNetB0 backbone
- Global Average Pooling
- Dense Layers
- Batch Normalization
- Dropout
- Softmax output layer

## Two-Stage Training Strategy

### Stage 1
- Backbone frozen
- Train only classifier head

### Stage 2
- Partial fine-tuning
- Top layers unfrozen
- BatchNorm layers kept frozen
- Lower learning rate used

## Fine-Tuning Experiments

Different unfreezing depths were tested:
- 30 layers
- 50 layers
- 120 layers

## Results

| Unfrozen Layers | Test Accuracy |
|---|---|
| 30 | 80.61% |
| 50 | 81.09% |
| 120 | 81.57% |

The results showed that deeper fine-tuning slightly improved domain adaptation, although the gains remained relatively small.

---

# 3. Patch-Based RNN

A sequence-based approach was explored using image patches.

## Method

- Images divided into `16×16` patches
- Total patches per image: `196`
- Each patch flattened into a sequence vector
- Bidirectional GRU used for sequential modeling

## Motivation

The goal was to investigate whether sequence modeling could capture relationships between image regions.

## Limitation

The patch-based representation weakened global spatial structure preservation, which significantly reduced performance in medical imaging tasks.

## Results

```python
Test Accuracy ≈ 61.86%
```

Despite experimenting with deeper architectures, performance improvements remained limited.

---

# Model Comparison

| Model | Test Accuracy |
|---|---|
| Custom CNN | 83.33% |
| EfficientNetB0 (Frozen) | 80.45% |
| EfficientNetB0 (Fine-Tuned) | 81.09% |
| Patch-Based RNN | 61.86% |

## Key Observation

Convolution-based architectures significantly outperformed the sequence-based approach because they preserve spatial relationships, which are critical in medical image analysis.

---

# Key Technical Learnings

Through this project we explored:

- CNN feature hierarchy
- Transfer learning strategies
- Fine-tuning tradeoffs
- Overfitting reduction techniques
- Class imbalance handling
- Spatial representation importance
- Medical imaging preprocessing
- Sequential vs convolutional architectures
- Training stabilization methods

---

# Technologies Used

```python
Python
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Scikit-learn
EfficientNetB0
GRU / RNN
```

---

# Repository Structure

```bash
Chest-X-Ray-Classification/
│
├── notebooks/
│   ├── Chest_X_Ray_Classification.ipynb
│   └── Test_Environment.ipynb
│
├── Project_Report.txt
├── README.md
```

---

# How to Run

## 1. Clone Repository

```bash
git clone https://github.com/AsherPe/Chest-X-Ray-Classification.git
```

## 2. Upload Notbook to Google colab or Jupyter 
run in Google Colab or Jupyter Notebook.

---

# Future Improvements

Possible future directions include:

- Vision Transformers (ViT)
- Attention-based CNN architectures
- Explainability methods (Grad-CAM)
- Self-supervised pretraining
- Advanced augmentation techniques
- Ensemble learning approaches

---

# Author

- Asher Pe’er

---

# License

This project was developed for educational and research purposes.
