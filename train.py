import argparse
import os
import pandas as pd
import joblib
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# Machine Learning Models
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

# ML Models
ML_MODELS = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'LightGBM': lgb.LGBMClassifier(),
    'Passive Aggressive Classifier': PassiveAggressiveClassifier(),
    'Perceptron': Perceptron(),
    'SGD Classifier': SGDClassifier(),
    'Stacking Classifier': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()), 
            ('gb', BaggingClassifier())
        ],
        final_estimator=LogisticRegression()
    ),
    'Voting Classifier (Hard)': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()), 
            ('gb', BaggingClassifier()), 
            ('svc', SVC(probability=True))
        ],
        voting='hard'
    ),
    'Voting Classifier (Soft)': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier()), 
            ('gb', BaggingClassifier()), 
            ('svc', SVC(probability=True))
        ],
        voting='soft'
    ),
}

# DL Models
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

# Argument Parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Train ML and/or DL models on text data.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--test_path", type=str, required=True, help="Path to testing CSV file.")
    parser.add_argument("--models", nargs="+", required=True, help="List of models to train (ML/DL).")
    parser.add_argument("--vectorizer", type=str, choices=["tfidf", "count", "hashing"], help="Choose vectorizer for ML models.")
    parser.add_argument("--max_len", type=int, default=128, help="Maximum sequence length for DL models.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DL models.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for DL models.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for DL models.")
    parser.add_argument("--output_dir", type=str, default="./models", help="Directory to save trained models.")
    return parser.parse_args()

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

# ML Training Function
def train_ml_model(model, vectorizer, train_df, test_df, output_path, vectorizer_name):
    X_train = vectorizer.fit_transform(train_df['crimeaditionalinfo'])
    X_test = vectorizer.transform(test_df['crimeaditionalinfo'])
    y_train = train_df['label']
    y_test = test_df['label']
    
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(output_path, f"{type(model).__name__}.pkl"))
    joblib.dump(vectorizer, os.path.join(output_path, f"{vectorizer_name}_vectorizer.pkl"))

    y_pred = model.predict(X_test)
    print_metrics(y_test, y_pred)

# DL Training Function
def train_dl_model(model_name, train_dataset, test_dataset, learning_rate, epochs, batch_size, output_path, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(DL_MODELS[model_name])
    model = AutoModelForSequenceClassification.from_pretrained(DL_MODELS[model_name], num_labels=num_labels)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Training"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['label'].to(model.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{epochs} Validation"):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['label'].to(model.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(test_loader)

        # Print metrics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print_metrics(all_labels, all_preds)

        # Save model if validation loss decreases
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(output_path, f"{model_name.replace('/', '_')}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

# Print Metrics
def print_metrics(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

# Main
def main():
    args = parse_arguments()
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    num_class = test_df['label'].unique().shape[0]
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in args.models:
        if model_name in ML_MODELS:
            vectorizer = {"tfidf": TfidfVectorizer, "count": CountVectorizer, "hashing": HashingVectorizer}[args.vectorizer]()
            train_ml_model(ML_MODELS[model_name], vectorizer, train_df, test_df, args.output_dir, args.vectorizer)
        elif model_name in DL_MODELS:
            tokenizer = AutoTokenizer.from_pretrained(DL_MODELS[model_name])
            train_dataset = CrimeDataset(train_df['crimeaditionalinfo'], train_df['label'], tokenizer, args.max_len)
            test_dataset = CrimeDataset(test_df['crimeaditionalinfo'], test_df['label'], tokenizer, args.max_len)
            train_dl_model(model_name, train_dataset, test_dataset, args.learning_rate, args.epochs, args.batch_size, args.output_dir, num_class)

if __name__ == "__main__":
    main()
