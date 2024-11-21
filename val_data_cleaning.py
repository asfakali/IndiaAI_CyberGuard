import pandas as pd
import numpy as np
import re
import argparse

# Define the category and label mappings
category_mapping = {
    "Any Other Cyber Crime": 2,
    "Cryptocurrency Crime": 5,
    "Cyber Attack/ Dependent Crimes": 1,
    "Cyber Terrorism": 8,
    "Hacking  Damage to computercomputer system etc": 3,
    "Online Cyber Trafficking": 7,
    "Online Financial Fraud": 0,
    "Online Gambling  Betting": 6,
    "Online and Social Media Related Crime": 4,
    "Ransomware": 9
}

subcategory_mapping = {
    "Business Email CompromiseEmail Takeover": 27,
    "Cheating by Impersonation": 8,
    "Cryptocurrency Fraud": 19,
    "Cyber Bullying  Stalking  Sexting": 17,
    "Cyber Terrorism": 29,
    "Damage to computer computer systems etc": 7,
    "Data Breach/Theft": 13,
    "DebitCredit Card FraudSim Swap Fraud": 0,
    "DematDepository Fraud": 23,
    "Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks": 22,
    "EMail Phishing": 11,
    "EWallet Related Fraud": 10,
    "Email Hacking": 15,
    "FakeImpersonating Profile": 14,
    "Fraud CallVishing": 2,
    "Hacking/Defacement": 18,
    "Impersonating Email": 30,
    "Internet Banking Related Fraud": 4,
    "Intimidating Email": 33,
    "Malware Attack": 9,
    "Online Gambling  Betting": 25,
    "Online Job Fraud": 16,
    "Online Matrimonial Fraud": 20,
    "Online Trafficking": 28,
    "Other": 3,
    "Profile Hacking Identity Theft": 12,
    "Provocative Speech for unlawful acts": 24,
    "Ransomware": 32,
    "Ransomware Attack": 26,
    "SQL Injection": 1,
    "Tampering with computer source documents": 21,
    "UPI Related Frauds": 6,
    "Unauthorised AccessData Breach": 5,
    "Website DefacementHacking": 31
}

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

def process_data(input_train, output_dir):
    # Load datasets
    train_df = pd.read_csv(input_train)
    

    # Drop missing values
    train_df = train_df.dropna()
    

    # Clean 'crimeaditionalinfo' column
    train_df['crimeaditionalinfo'] = train_df['crimeaditionalinfo'].apply(clean_text)
    

    # Process by sub_category
    train_filtered = train_df[train_df['sub_category'].isin(subcategory_mapping.keys())]
    train_filtered = train_filtered.drop(columns=['category'])

    # Map subcategory to label
    train_filtered['label'] = train_filtered['sub_category'].map(subcategory_mapping)

    train_filtered = train_filtered.dropna(axis=0, how='any')

    # Save subcategory results
    train_filtered.to_csv(f'{output_dir}/val_sub_category.csv', index=False)

    # Process by category
    train_filtered = train_df[train_df['category'].isin(category_mapping.keys())]
    train_filtered = train_filtered.drop(columns=['sub_category'])

    # Map category to label
    train_filtered['label'] = train_filtered['category'].map(category_mapping)

    train_filtered = train_filtered.dropna(axis=0, how='any')


    # Save category results
    train_filtered.to_csv(f'{output_dir}/val_category.csv', index=False)

    print(f"Processing complete. Files saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and process datasets for categories and subcategories.")
    parser.add_argument('--input_val', type=str, required=True, help="Path to the Validation dataset CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the processed files.")
    args = parser.parse_args()

    process_data(args.input_val, args.output_dir)
