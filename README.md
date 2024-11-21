

# IndiaAI_CyberGuard

## Data Cleaning and Processing Script

This repository provides a Python script for cleaning and processing datasets in preparation for classification tasks. The script handles data cleaning by removing unnecessary elements from text, processes the data based on `category` and `sub_category`, and generates labeled CSV files for both training and testing.

## Features

- **Text Data Cleaning**: Removes numbers, URLs, punctuation, and extra whitespace from the text.
- **Dataset Processing**: Filters data to only include overlapping `categories` and `sub_categories` between training and testing datasets.
- **Labeling**: Generates labeled datasets for both `category` and `sub_category` for easy integration into machine learning models.
- **Output**: Saves the cleaned and labeled datasets as CSV files for further processing.

## Requirements

To run the script, you'll need Python 3.7 or higher and the following libraries:

- `pandas`
- `numpy`
- `re` (Python built-in)

To install the required libraries, use the following command:

```bash
pip install pandas numpy
```

## Usage

### 1. Training and Testing Data Cleaning

Run the `data_cleaning.py` script to process the training and testing datasets:

```bash
python data_cleaning.py --input_train <path_to_train_csv> --input_test <path_to_test_csv> --output_dir <output_directory>
```

**Example:**

```bash
python data_cleaning.py --input_train ./train.csv --input_test ./test.csv --output_dir ./processed_data
```

This will clean the data, process categories and subcategories, and save the results in the specified output directory.

### 2. Validation Data Cleaning

To clean validation data separately, run the `val_clean.py` script:

```bash
python val_data_cleaning.py --input_val <path_to_val_csv> --output_dir <output_directory>
```

**Example:**

```bash
python val_data_cleaning.py --input_val val.csv --output_dir ./processed_data
```

This will clean the validation data and save the cleaned dataset in the specified output directory.

## Example Directory Structure

```
/IndiaAI_CyberGuard
  ├── /processed_data           # Folder where cleaned CSV files are saved
  ├── /model_file
      ├── /ml_models
      ├── /dl_model
      ├── /vectorizer
  ├── data_cleaning.py          # Script for cleaning train and test data
  ├── val_clean.py              # Script for cleaning validation data
  ├── README.md                 # This file
  └── requirements.txt          # List of dependencies
```

# Text Classification with ML and DL Models

This repository contains a Python script to train both Machine Learning (ML) and Deep Learning (DL) models on text data for classification tasks. It supports a wide range of ML algorithms and various DL models, including pre-trained transformers like BERT, Electra, and GPT2. The script allows the use of different vectorization methods for the ML models and provides an easy-to-follow pipeline for training, evaluation, and model saving.

## Features

- **Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and more.
- **Deep Learning Models**: Pre-trained transformers like BERT, GPT2, MobileBERT, and others.
- **Text Vectorization**: TF-IDF, Count Vectorizer, and Hashing Vectorizer for ML models.
- **Metrics**: Accuracy, Precision, Recall, F1 Score.
- **Model Saving**: Saves the best models based on validation loss during training.
- **Customizable**: Choose vectorizers, models, batch size, learning rate, and more via command-line arguments.

## Requirements

To run the script, you'll need Python 3.7 or higher and the following libraries:

- `pandas`
- `numpy`
- `torch`
- `transformers`
- `sklearn`
- `lightgbm`
- `joblib`
- `tqdm`

Install the required libraries with the following command:

```bash
pip install pandas numpy torch transformers scikit-learn lightgbm joblib tqdm

```

## Usage

### 1. Train ML and DL Models

Run the `train_models.py` script to train the selected models on the given training and testing data:

```bash
python train_models.py --train_path <path_to_train_csv> --test_path <path_to_test_csv> --models <model1> <model2> ... --vectorizer <tfidf/count/hashing> --output_dir <output_directory>
```

**Example:**

```bash
python train_models.py --train_path ./train.csv --test_path ./test.csv --models Logistic\ Regression SVM --vectorizer tfidf --output_dir ./models
```

This will train the specified ML models (`Logistic Regression`, `SVM`) using the `TF-IDF` vectorizer and save the trained models in the `./models` directory.

### 2. Customization Options

- **train_path**: Path to the training dataset (CSV file).
- **test_path**: Path to the testing dataset (CSV file).
- **models**: List of models to train (choose from ML or DL models, e.g., `Logistic Regression`, `Random Forest`, `BERT`).
- **vectorizer**: Choose vectorizer for ML models (`tfidf`, `count`, `hashing`).
- **max_len**: Maximum sequence length for DL models (default is 128).
- **batch_size**: Batch size for DL models (default is 32).
- **learning_rate**: Learning rate for DL models (default is 2e-5).
- **epochs**: Number of epochs for DL models (default is 10).
- **output_dir**: Directory to save trained models (default is `./models`).

### 3. Supported ML Models

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

### 4. Supported DL Models

- `google/electra-small-discriminator`
- `squeezebert/squeezebert-uncased`
- `google/mobilebert-uncased`
- `huawei-noah/TinyBERT_General_4L_312D`
- `microsoft/MiniLM-L12-H384-uncased`
- `google/bigbird-roberta-base`
- `gpt2`
- `xlnet-base-cased`
- `bert-base-uncased`

### 5. Metrics Printed During Training

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### 6. Model Saving

The script saves the best models based on the validation loss after each epoch. The saved models can be found in the `output_dir` folder.

## Example Directory Structure

```
/text-classification
  ├── /models                # Folder where trained models are saved
  ├── train_models.py        # Script to train ML and DL models
  ├── README.md              # This file
  ├── requirements.txt       # List of dependencies
  ├── train.csv              # Example training dataset (CSV)
  ├── test.csv               # Example testing dataset (CSV)
  └── output/                # Folder for model output and logs
```

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more information.
```

This README provides a detailed guide for running and understanding the script, including customization options and the supported models.

