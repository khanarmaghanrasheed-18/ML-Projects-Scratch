# Breast Cancer Prediction Pipeline From Scratch

An end-to-end machine learning pipeline built from scratch using **Python** and **NumPy** to classify breast tumors as **malignant** or **benign**.

The project combines:

- Manual feature standardization
- Principal Component Analysis (PCA) from scratch
- Linear Support Vector Machine (SVM) from scratch
- A custom Pipeline class
- PCA and SVM visualizations
- Evaluation on the Breast Cancer Wisconsin dataset

## Project Goal

The goal was not only to train a classifier, but to understand what happens behind common machine learning methods such as:

```python
.fit()
.transform()
.predict()
```

Instead of treating these methods as black boxes, each stage of the workflow was implemented manually.

## Dataset

The project uses the Breast Cancer Wisconsin dataset available through `scikit-learn`.

- Samples: 569
- Original features: 30
- Target classes:
  - `-1` → Malignant
  - `+1` → Benign

The data is split into training and testing sets before preprocessing.

## Pipeline

```text
Breast Cancer Dataset
        ↓
Train/Test Split
        ↓
StandardScaler
        ↓
PCA: 30 Features → 2 Principal Components
        ↓
Linear SVM
        ↓
Malignant / Benign Prediction
```

## Components Implemented

### 1. StandardScaler

The scaler learns the mean and standard deviation of every feature from the training data.

```text
fit()       → learns mean and standard deviation
transform() → scales data using the learned values
```

The same training statistics are used to transform both the training and test sets, preventing data leakage.

### 2. Principal Component Analysis

PCA reduces the original 30-dimensional feature space to two principal components.

The implementation includes:

- Mean centering
- Covariance matrix construction
- Eigenvalue and eigenvector calculation
- Sorting principal directions by explained variance
- Projection onto the top two components

```text
fit()       → learns the principal component matrix W
transform() → projects new data into PCA space
```

The two principal components are weighted combinations of all 30 original medical features.

### 3. Linear Support Vector Machine

A linear SVM was trained using stochastic gradient descent and hinge-loss-based updates.

The implementation includes:

- Weight and bias initialization
- Margin condition
- Regularization
- Gradient updates
- Binary predictions using the decision function

The classifier predicts:

```text
M → Malignant
B → Benign
```

### 4. Custom Pipeline

The custom Pipeline class connects all components into one reusable workflow.

```python
pipeline = Pipeline(
    StandardScaler(),
    PCA(final=2),
    SVM()
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Internally, the pipeline performs:

```text
Scaler.fit()
Scaler.transform()
PCA.fit()
PCA.transform()
SVM.fit()
```

During prediction, it only applies the transformations already learned from the training data:

```text
Scaler.transform()
PCA.transform()
SVM.predict()
```

## Results

The final pipeline achieved approximately:

```text
Test Accuracy: 95%
```

The exact result may vary slightly depending on the optimizer settings and random split.

## Visualization

The project produces two plots in one window:

1. Breast cancer samples projected onto Principal Component 1 and Principal Component 2
2. Linear SVM decision boundary, margins, and approximate support vectors in PCA space

The visualization makes it possible to observe how PCA and SVM work together:

- PCA compresses the original 30 features into two dimensions
- SVM learns a decision boundary in the reduced feature space

## Why PCA Before SVM?

The original dataset contains 30 features, which cannot be directly visualized.

PCA reduces the feature space to two dimensions while preserving as much variance as possible. The SVM can then be trained and visualized in this reduced space.

This demonstrates how two machine learning algorithms can work together in a complete pipeline rather than being treated as isolated topics.

## Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn only for loading the dataset and creating the train/test split

## Key Learning

The biggest takeaway from this project was understanding that methods such as `.fit()`, `.transform()`, and `.predict()` are not magic.

They represent separate stages:

```text
fit()       → learn parameters
transform() → apply learned transformations
predict()   → produce final outputs
```

Building each stage from scratch made the full machine learning pipeline much easier to understand and removed the black-box feeling behind common library calls.

## Future Improvements

- Implement GridSearchCV from scratch
- Add cross-validation
- Tune learning rate and regularization strength
- Compare 2-component PCA accuracy with SVM trained on all 30 features
- Add explained variance ratio
- Add automatic support-vector detection
- Implement kernel SVM