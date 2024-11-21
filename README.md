# README for IndiaAI_CyberGuard

## Overview

The **IndiaAI_CyberGuard** repository provides tools for data preprocessing and text classification. It includes scripts for cleaning datasets and training both Machine Learning (ML) and Deep Learning (DL) models, supporting various vectorization techniques and pre-trained transformers for effective classification.

## Text Classification Models

### Features

- Supports both ML (e.g., Logistic Regression, Random Forest) and DL (e.g., BERT, GPT2) models.
- Text vectorization via TF-IDF, Count Vectorizer, or Hashing Vectorizer.
- Customizable parameters: batch size, learning rate, epochs, etc.
- Logs key metrics: Accuracy, Precision, Recall, F1 Score.
- Saves best-performing models based on validation performance.

---

## Table of Contents

1. [Data Cleaning and Processing](#data-cleaning-and-processing)
   - [Features](#features)
   - [Requirements](#requirements)
   - [Usage](#usage)
   - [Example Directory Structure](#example-directory-structure)
2. [Text Classification Models](#text-classification-models)
   - [Usage](#usage-1)
   - [Supported Models](#supported-models)
   - [Metrics and Model Saving](#metrics-and-model-saving)
3. [License](#license)

---

## Data Cleaning and Processing

### Features

- **Text Cleaning**: Removes unwanted elements like numbers, URLs, punctuation, and whitespace.
- **Category Filtering**: Retains overlapping `categories` and `sub_categories` across datasets.
- **Labeled Datasets**: Prepares labeled CSV files for classification tasks.
- **Output Files**: Saves processed datasets for downstream tasks.

### Requirements

- Python 3.7 or higher
- Libraries: `pandas`, `numpy`

Install required libraries:

```bash
pip install -r 
```

### Usage

#### Training and Testing Data Cleaning

Use `data_cleaning.py` to clean and process datasets:

```bash
python data_cleaning.py --input_train <path_to_train_csv> --input_test <path_to_test_csv> --output_dir <output_directory>
```

Example:

```bash
python data_cleaning.py --input_train ./train.csv --input_test ./test.csv --output_dir ./processed_data
```

#### Validation Data Cleaning

Use `val_clean.py` to process validation data:

```bash
python val_clean.py --input_val <path_to_val_csv> --output_dir <output_directory>
```

Example:

```bash
python val_clean.py --input_val ./val.csv --output_dir ./processed_data
```

### Example Directory Structure

```
/IndiaAI_CyberGuard
  ├── /processed_data        # Folder for cleaned datasets
  ├── /model_file
      ├── /ml_models
      ├── /dl_model
      ├── /vectorizer
  ├── data_cleaning.py       # Train and test data cleaning script
  ├── val_clean.py           # Validation data cleaning script
  ├── README.md              # Documentation
  └── requirements.txt       # Dependencies
```

---



### Usage

Train models using `train_models.py`:

```bash
python train_models.py --train_path <path_to_train_csv> --test_path <path_to_test_csv> --models <model1> <model2> ... --vectorizer <vectorizer_type> --output_dir <output_directory>
```

Example:

```bash
python train_models.py --train_path ./train.csv --test_path ./test.csv --models Logistic\ Regression SVM --vectorizer tfidf --output_dir ./models
```

#### Customization Options

- **train_path**: Path to training dataset.
- **test_path**: Path to testing dataset.
- **models**: Specify ML or DL models (e.g., `Logistic Regression`, `BERT`).
- **vectorizer**: Vectorization method (`tfidf`, `count`, `hashing`).
- **max_len**: Max sequence length for DL models (default: 128).
- **batch_size**: Batch size for DL models (default: 32).
- **learning_rate**: Learning rate for DL models (default: 2e-5).
- **epochs**: Number of epochs for DL models (default: 10).
- **output_dir**: Directory for saving trained models.

---

### Supported Models

#### Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- LightGBM
- AdaBoost, and more.

#### Deep Learning
- Pre-trained transformers: `bert-base-uncased`, `gpt2`, `google/mobilebert-uncased`, etc.

---

### Metrics and Model Saving

- Logs Accuracy, Precision, Recall, and F1 Score during training.
- Saves best models in the specified `output_dir` based on validation loss.

