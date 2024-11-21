

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
python val_clean.py --input_val <path_to_val_csv> --output_dir <output_directory>
```

**Example:**

```bash
python val_clean.py --input_val val.csv --output_dir ./processed_data
```

This will clean the validation data and save the cleaned dataset in the specified output directory.

## Example Directory Structure

```
/IndiaAI_CyberGuard
  ├── /processed_data           # Folder where cleaned CSV files are saved
  ├── data_cleaning.py          # Script for cleaning train and test data
  ├── val_clean.py              # Script for cleaning validation data
  ├── README.md                 # This file
  └── requirements.txt          # List of dependencies
```

## License

This repository is licensed under the MIT License. See the `LICENSE` file for more information.

