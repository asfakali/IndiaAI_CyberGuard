# IndiaAI_CyberGuard

# Data Cleaning and Processing Script

This repository contains a Python script for cleaning and processing datasets for classification tasks. The script processes data based on `category` and `sub_category`, cleans textual data, and generates labeled CSV files for training and testing.

## Features

- Cleans text data by removing numbers, URLs, punctuation, and extra whitespace.
- Processes datasets to include only overlapping `categories` and `sub_categories` between training and testing data.
- Generates labeled datasets for both `category` and `sub_category`.
- Saves the cleaned datasets as CSV files.

## Requirements

- Python 3.7 or higher
- Pandas
- NumPy
- re (built-in)

Install the required libraries using:

```bash
pip install pandas numpy
