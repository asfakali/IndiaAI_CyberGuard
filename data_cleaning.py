import pandas as pd
import numpy as np
import re
import argparse

# Function to clean text
def clean_text(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespaces
    text = " ".join(text.split())
    
    return text if text != '' else np.nan

def process_data(input_train, input_test, output_dir):
    # Load datasets
    train_df = pd.read_csv(input_train)
    test_df = pd.read_csv(input_test)

    # Drop missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Get unique categories and subcategories
    train_cat = set(train_df['category'].unique())
    test_cat = set(test_df['category'].unique())
    train_subcat = set(train_df['sub_category'].unique())
    test_subcat = set(test_df['sub_category'].unique())

    # Clean 'crimeaditionalinfo' column
    train_df['crimeaditionalinfo'] = train_df['crimeaditionalinfo'].apply(clean_text)
    test_df['crimeaditionalinfo'] = test_df['crimeaditionalinfo'].apply(clean_text)

    # Process by sub_category
    train_filtered = train_df[train_df['sub_category'].isin(train_subcat & test_subcat)]
    test_filtered = test_df[test_df['sub_category'].isin(train_subcat & test_subcat)]
    train_filtered = train_filtered.drop(columns=['category'])
    test_filtered = test_filtered.drop(columns=['category'])
    categories_list = list(test_filtered['sub_category'].unique())
    categories_list.sort()
    train_filtered['label'] = train_filtered['sub_category'].map(lambda x: categories_list.index(x) if x in categories_list else -1)
    test_filtered['label'] = test_filtered['sub_category'].map(lambda x: categories_list.index(x) if x in categories_list else -1)
    train_filtered = train_filtered.dropna(axis=0, how='any')
    test_filtered = test_filtered.dropna(axis=0, how='any')
    train_filtered.to_csv(f'{output_dir}/train_sub_category.csv', index=False)
    test_filtered.to_csv(f'{output_dir}/test_sub_category.csv', index=False)

    # Process by category
    train_filtered = train_df[train_df['category'].isin(train_cat & test_cat)]
    test_filtered = test_df[test_df['category'].isin(train_cat & test_cat)]
    train_filtered = train_filtered.drop(columns=['sub_category'])
    test_filtered = test_filtered.drop(columns=['sub_category'])
    categories_list = list(test_filtered['category'].unique())
    categories_list.sort()
    train_filtered['label'] = train_filtered['category'].map(lambda x: categories_list.index(x) if x in categories_list else -1)
    test_filtered['label'] = test_filtered['category'].map(lambda x: categories_list.index(x) if x in categories_list else -1)
    train_filtered = train_filtered.dropna(axis=0, how='any')
    test_filtered = test_filtered.dropna(axis=0, how='any')
    train_filtered.to_csv(f'{output_dir}/train_category.csv', index=False)
    test_filtered.to_csv(f'{output_dir}/test_category.csv', index=False)

    print(f"Processing complete. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and process datasets for categories and subcategories.")
    parser.add_argument('--input_train', type=str, required=True, help="Path to the training dataset CSV file.")
    parser.add_argument('--input_test', type=str, required=True, help="Path to the testing dataset CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed files.")
    args = parser.parse_args()

    process_data(args.input_train, args.input_test, args.output_dir)
