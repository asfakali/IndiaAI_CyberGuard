import os
import pandas as pd
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import signal
import sys
import warnings
warnings.filterwarnings('ignore')
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import logging
logging.set_verbosity_error()


# Define available DL models
DL_MODELS = {
    'Electra': 'google/electra-small-discriminator',
    'SqueezeBERT': 'squeezebert/squeezebert-uncased',
    'MobileBERT': 'google/mobilebert-uncased',
    'TinyBERT': 'huawei-noah/TinyBERT_General_4L_312D',
    'MiniLM': 'microsoft/MiniLM-L12-H384-uncased',
    'BigBird': 'google/bigbird-roberta-base',
    'GPT2': 'gpt2',
    'XLNet': 'xlnet-base-cased',
    'BERT': 'bert-base-uncased',
}


category_mapping= {
    2: "Any Other Cyber Crime",
    5: "Cryptocurrency Crime",
    1: "Cyber Attack/ Dependent Crimes",
    8: "Cyber Terrorism",
    3: "Hacking  Damage to computercomputer system etc",
    7: "Online Cyber Trafficking",
    0: "Online Financial Fraud",
    6: "Online Gambling  Betting",
    4: "Online and Social Media Related Crime",
    9: "Ransomware"
}

subcategory_mapping = {
    27: "Business Email CompromiseEmail Takeover",
    8: "Cheating by Impersonation",
    19: "Cryptocurrency Fraud",
    17: "Cyber Bullying  Stalking  Sexting",
    29: "Cyber Terrorism",
    7: "Damage to computer computer systems etc",
    13: "Data Breach/Theft",
    0: "DebitCredit Card FraudSim Swap Fraud",
    23: "DematDepository Fraud",
    22: "Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks",
    11: "EMail Phishing",
    10: "EWallet Related Fraud",
    15: "Email Hacking",
    14: "FakeImpersonating Profile",
    2: "Fraud CallVishing",
    18: "Hacking/Defacement",
    30: "Impersonating Email",
    4: "Internet Banking Related Fraud",
    33: "Intimidating Email",
    9: "Malware Attack",
    25: "Online Gambling  Betting",
    16: "Online Job Fraud",
    20: "Online Matrimonial Fraud",
    28: "Online Trafficking",
    3: "Other",
    12: "Profile Hacking Identity Theft",
    24: "Provocative Speech for unlawful acts",
    32: "Ransomware",
    26: "Ransomware Attack",
    1: "SQL Injection",
    21: "Tampering with computer source documents",
    6: "UPI Related Frauds",
    5: "Unauthorised AccessData Breach",
    31: "Website DefacementHacking"
}


# Dataset class for DL models
class CrimeDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


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
    
    return text
    
    
# Prediction for individual data
def predict_ml(model_path, vectorizer_path, text):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    vectorized_text = vectorizer.transform([clean_text(text)])
    prediction = model.predict(vectorized_text)
    return prediction[0]

def predict_dl(model_name, model_path, text, max_len, num_labels=2):
    tokenizer = AutoTokenizer.from_pretrained(DL_MODELS[model_name])
    try:
    	model = AutoModelForSequenceClassification.from_pretrained(DL_MODELS[model_name], num_labels=2)
    	model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    except:
    	model = AutoModelForSequenceClassification.from_pretrained(DL_MODELS[model_name], num_labels=34)
    	model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    
    model.eval()

    dataset = CrimeDataset([clean_text(text)], tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
    return prediction

# Signal handler to prevent termination on interrupt
def handle_interrupt(signal, frame):
    print("\nProcess is running in quiet mode. Use Ctrl+C again to terminate.")
    sys.exit(0)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Predict individual data using a selected ML or DL model.")
    parser.add_argument('--model_type', type=str, choices=['ml', 'dl'], required=True, help="Type of model to use (ml or dl).")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to use.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model file.")
    parser.add_argument('--vectorizer_path', type=str, required=False, help="Path to the vectorizer file (required for ML models).")
    parser.add_argument('--max_len', type=int, default=128, help="Max sequence length for DL models.")
    parser.add_argument('--categorical', type=str, choices=['categorical', 'subcategorical'], required=True, help="Type of label categorical or subcategorical.")
    args = parser.parse_args()

    # Ensure vectorizer path for ML models
    if args.model_type == 'ml' and not args.vectorizer_path:
        print("Vectorizer path is required for ML models.")
        sys.exit(1)

    if args.model_type == 'dl' and args.model_name not in DL_MODELS:
        print(f"Model {args.model_name} is not supported.")
        sys.exit(1)

    print("Type 'Exit' to quit the program.\n")

    while True:
        user_input = input("Enter text for prediction: ")
        if user_input.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break
        
        # Make prediction
        if args.model_type == 'ml':
            prediction = predict_ml(args.model_path, args.vectorizer_path, user_input)
        elif args.model_type == 'dl':
            prediction = predict_dl(args.model_name, args.model_path, user_input, args.max_len)
        else:
            print("Invalid model type selected.")
            continue

        # Map prediction to category or subcategory
        if args.categorical == 'categorical':
            prediction_mapped = category_mapping.get(prediction, "Unknown Category")
            print(f"\nPredicted Category: {prediction_mapped}\n")
        else:
            prediction_mapped = subcategory_mapping.get(prediction, "Unknown Subcategory")
            print(f"\nPredicted Subcategory: {prediction_mapped}\n")

if __name__ == "__main__":
    main()
