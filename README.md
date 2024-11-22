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
3. [Text Classification Pipline](#text-classification-pipline)
   - [Usage](#usage-2)
---

## Data Cleaning and Processing

### Features

- **Text Cleaning**: Removes unwanted elements like numbers, URLs, punctuation, and whitespace.
- **Category Filtering**: Retains overlapping `categories` and `sub_categories` across datasets.
- **Labeled Datasets**: Prepares labeled CSV files for classification tasks.
- **Output Files**: Saves processed datasets for downstream tasks.

### Requirements

- Python 3.7 or higher
- Libraries: `pandas`, `numpy`, `Pandas`, `NumPy`, `Scikit-learn`, `PyTorch`, `Transformers` (Hugging Face), `Matplotlib`, `Seaborn`, `TQDM`, `Joblib`, `json`

Install required libraries:

```bash
pip install -r requirements.txt
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

Use `val_data_cleaning.py` to process validation data:

```bash
python val_data_cleaning.py --input_val <path_to_val_csv> --output_dir <output_directory>
```

Example:

```bash
python val_data_cleaning.py --input_val ./val.csv --output_dir ./processed_data
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

Train models using `train.py`:

```bash
python train.py --train_path <path_to_train_csv> --test_path <path_to_test_csv> --models <model1> <model2> ... --vectorizer <vectorizer_type> --output_dir <output_directory>
```

Example:

```bash
python train.py --train_path ./train.csv --test_path ./test.csv --models Logistic\ Regression SVM --vectorizer tfidf --output_dir ./models_dir
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

#### Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- SVM (Support Vector Machine)
- KNN (K-Nearest Neighbors)
- AdaBoost
- Bagging
- Extra Trees
- LightGBM
- Passive Aggressive Classifier
- Perceptron
- SGD Classifier
- Stacking Classifier
- Voting Classifier (Hard and Soft)


#### Supported Deep Learning Models
- Pre-trained transformers: `google/electra-small-discriminator`, `squeezebert/squeezebert-uncased`, `huawei-noah/TinyBERT_General_4L_312D`, `microsoft/MiniLM-L12-H384-uncased`, `bert-base-uncased`, `gpt2`, `google/bigbird-roberta-base`, `xlnet-base-cased`, `google/mobilebert-uncased`.

---

### Metrics and Model Saving

- Logs Accuracy, Precision, Recall, and F1 Score during training.
- Saves best models in the specified `output_dir` based on validation loss.


Here’s a README file for your script:

---

# Text Classification Pipline

## Usage

### Command-line Arguments
The script takes the following arguments:

| Argument           | Required | Description                                                                                   |
|---------------------|----------|-----------------------------------------------------------------------------------------------|
| `--model_type`      | Yes      | Type of model to use: `ml` or `dl`.                                                           |
| `--model_name`      | Yes      | Name of the DL model (e.g., `google/mobilebert-uncased`).                                      |
| `--model_path`      | Yes      | Path to the saved model file.                                                                 |
| `--vectorizer_path` | No       | Path to the vectorizer file (required for ML models).                                         |
| `--max_len`         | No       | Max sequence length for DL models. Default is `128`.                                          |
| `--categorical`     | Yes      | Type of prediction: `categorical` for main categories or `subcategorical` for subcategories.  |

### Example Commands
#### Predict using a deep learning model:
```bash
python pipline.py --model_type dl --model_name google/mobilebert-uncased --model_path model.pt --categorical categorical
```

#### Predict using a machine learning model:
```bash
python script.py --model_type ml --model_path ml_model.pkl --vectorizer_path vectorizer.pkl --categorical subcategorical
```

### Running the Script
1. Start the script with the appropriate arguments.
2. Input text for prediction when prompted.
3. Type `Exit` to close the script.

### Sample Session
```bash
$ python script.py --model_type dl --model_name google/mobilebert-uncased --model_path model.pt --categorical categorical
Type 'Exit' to quit the program.

Enter text for prediction: Cybercrime involving financial fraud
Prediction Category: Online Financial Fraud

Enter text for prediction: Social media account hacking
Prediction Category: Online and Social Media Related Crime

Enter text for prediction: Exit
Exiting the program. Goodbye!
```

# Model Testing Framework

## Overview

This project provides a framework to test machine learning (ML) and deep learning (DL) models for text classification tasks. It supports various DL architectures and calculates key performance metrics such as accuracy, precision, recall, F1-score, and visualizes results using confusion matrices and ROC curves.

## Features

- **Machine Learning (ML) Support**:
  - Test pre-trained ML models using vectorized text data.
  - Compute performance metrics: accuracy, precision, recall, F1-score.
  - Generate confusion matrix visualizations.



## Usage

### Command-Line Arguments

Run the script using the following arguments:

| Argument           | Description                                                                                  | Default       |
|--------------------|----------------------------------------------------------------------------------------------|---------------|
| `--test_csv`       | Path to the test CSV file (required).                                                        | -             |
| `--ml_model_paths` | Paths to ML models to test.                                                                  | None          |
| `--dl_model_paths` | Dictionary of DL model names and paths (`model_name:model_path`).                            | None          |
| `--vectorizer_path`| Path to the vectorizer used by ML models.                                                    | None          |
| `--save_dir`       | Directory to save metrics and visualizations.                                                | -             |
| `--max_len`        | Maximum token length for DL models.                                                          | 128           |
| `--batch_size`     | Batch size for DL models.                                                                    | 16            |

### Example Command
```bash
python main.py \
  --test_csv data/test.csv \
  --ml_model_paths models/ml_model_1.pkl models/ml_model_2.pkl \
  --dl_model_paths BERT:models/bert_model.pth XLNet:models/xlnet_model.pth \
  --vectorizer_path models/vectorizer.joblib \
  --save_dir results/ \
  --max_len 128 \
  --batch_size 16
```

### Input Format
The test CSV should have the following columns:
- `crimeaditionalinfo`: The text input for classification.
- `label`: The true labels for evaluation.

### Output
1. **Performance Metrics**: 
   - Accuracy
   - Precision
   - Recall
   - F1-score

2. **Visualizations**:
   - Confusion Matrix: Saved as PNG files in the specified save directory.

### Example Outputs
- `results/bert_model_cm.png`: Confusion matrix for the BERT model.
- `results/ml_model_1_metrics.json`: Metrics for an ML model.

- Ensure `test_csv` matches the expected format.
- Check model paths and ensure correct formats (e.g., `.pkl` for ML, `.pth` for DL).
- Verify that required transformers and tokenizers are downloaded.

---

Happy Testing! 🎉
