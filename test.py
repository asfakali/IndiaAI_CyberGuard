import os
import pandas as pd
import joblib
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import sys
import warnings
warnings.filterwarnings('ignore')
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import logging
logging.set_verbosity_error()



# Define DL Models
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

# Dataset Class for DL Models
class CrimeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Save Metrics and Visualizations
def save_cm_as_png(y_true, y_pred, cm_filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(cm_filename)
    plt.close()

def save_auc_as_png(y_true, y_proba, auc_filename):
    if len(np.unique(y_true)) == 2:  # Binary classification
        auc = roc_auc_score(y_true, y_proba[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(auc_filename)
        plt.close()

# Test ML Model
def test_ml_model(model_path, vectorizer_path, test_df, cm_filename=None):
    print(f"Testing ML Model: {os.path.basename(model_path)}")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X_test = vectorizer.transform(test_df['crimeaditionalinfo'])
    y_test = test_df['label']
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Save Confusion Matrix
    if cm_filename:
        save_cm_as_png(y_test, y_pred, cm_filename)
    
    print(f"Metrics for {os.path.basename(model_path)}: {metrics}")
    return metrics

# Test DL Model
def test_dl_model(model_name, model_path, test_dataset, batch_size, cm_filename=None, num_labels=2):
    print(f"Testing DL Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(DL_MODELS[model_name])
    try:
    	model = AutoModelForSequenceClassification.from_pretrained(DL_MODELS[model_name], num_labels=2)
    	model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    except:
    	model = AutoModelForSequenceClassification.from_pretrained(DL_MODELS[model_name], num_labels=34)
    	model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Save Confusion Matrix
    if cm_filename:
        save_cm_as_png(all_labels, all_preds, cm_filename)

    print(f"Metrics for {model_name}: {metrics}")
    return metrics

# Main
def main(test_csv, ml_model_paths, dl_model_paths, vectorizer_path, max_len, batch_size, save_dir):
    test_df = pd.read_csv(test_csv)
    num_labels = len(test_df['label'].unique())

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Test ML Models
    for model_path in ml_model_paths:
        cm_filename = os.path.join(save_dir, f"{os.path.basename(model_path)}_cm.png")
        test_ml_model(model_path, vectorizer_path, test_df, cm_filename)

    # Test DL Models
    for model_name, model_path in dl_model_paths.items():
        tokenizer = AutoTokenizer.from_pretrained(DL_MODELS[model_name])
        test_dataset = CrimeDataset(
            test_df['crimeaditionalinfo'], 
            test_df['label'], 
            tokenizer, 
            max_len
        )
        cm_filename = os.path.join(save_dir, f"{model_name}_cm.png")
        test_dl_model(model_name, model_path, test_dataset, batch_size, cm_filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test ML/DL models")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test CSV")
    parser.add_argument('--ml_model_paths', type=str, nargs='*', required=False, help="Paths to ML model files")
    parser.add_argument('--dl_model_paths', type=str, nargs='+', required=False, help="DL model names and paths as 'model_name:model_path'")
    parser.add_argument('--vectorizer_path', type=str, required=False, help="Path to the vectorizer")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the results")
    parser.add_argument('--max_len', type=int, default=128, help="Max sequence length for DL models")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for DL models")
    args = parser.parse_args()

    # Parse DL model paths as a dictionary
    dl_model_paths = dict(
        pair.split(":") for pair in args.dl_model_paths
    ) if args.dl_model_paths else {}

    main(
        args.test_csv,
        args.ml_model_paths or [],
        dl_model_paths,
        args.vectorizer_path,
        args.max_len,
        args.batch_size,
        args.save_dir
    )

