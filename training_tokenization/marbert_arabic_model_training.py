import os
import pandas as pd
from lxml import etree
from tqdm import tqdm
import re
import numpy as np
import torch
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# === Configuration ===
class Config:
    SEED = 42
    DATA_DIR = "/content/drive/MyDrive/ar"
    MODEL_NAME = "UBC-NLP/MARBERT"
    OUTPUT_DIR = "./models/marbert_arabic_pan"
    MAX_LENGTH = 128
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 6
    EARLY_STOPPING_PATIENCE = 2
    RESULT_LOG = "arabic_marbert_evaluation.csv"
    PREDICTION_LOG = "arabic_marbert_predictions.csv"
    DIALECT_TAGS = {'lev': '[LEV]', 'glf': '[GLF]', 'egy': '[EGY]', 'nor': '[NOR]', 'irq': '[IRQ]'}

# === Setup ===
def setup_environment():
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed_all(Config.SEED)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Text Cleaning ===
class ArabicTextCleaner:
    @staticmethod
    def clean(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'https?://\S+|www\.\S+|@\w+|\#\w+', '', text)
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'[يى]', 'ي', text)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652ـ]', '', text)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

# === Data Loading ===
def load_and_process_data():
    user_gender = {}
    with open(os.path.join(Config.DATA_DIR, "truth.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) >= 2:
                user_gender[parts[0]] = 0 if parts[1] == "male" else 1

    data = []
    parser = etree.XMLParser(encoding="utf-8", recover=True)
    for filename in tqdm(os.listdir(Config.DATA_DIR), desc="Processing XML files"):
        if filename.endswith(".xml"):
            user_id = filename[:-4]
            file_path = os.path.join(Config.DATA_DIR, filename)
            try:
                tree = etree.parse(file_path, parser=parser)
                posts = [p.strip() for p in tree.xpath("//document/text()") if p.strip()]
                if posts and user_id in user_gender:
                    full_text = ArabicTextCleaner.clean(" ".join(posts))
                    if full_text:
                        data.append({"text": full_text, "label": user_gender[user_id]})
            except Exception as e:
                print(f"❌ Error parsing {filename}: {e}")
    return pd.DataFrame(data)

# === Tokenization ===
def tokenize_and_prepare_dataset(df):
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = Dataset.from_pandas(df)

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset = dataset.train_test_split(test_size=0.1, seed=Config.SEED)
    return dataset, tokenizer

# === Metrics ===
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    p_score, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    cm = confusion_matrix(labels, preds)
    print("📊 Confusion Matrix:\n", cm)
    print("📋 Classification Report:\n", classification_report(labels, preds))
    
    metrics_df = pd.DataFrame([{
        "accuracy": acc,
        "precision": p_score,
        "recall": r,
        "f1": f1,
        "confusion_matrix": cm.tolist()
    }])
    metrics_df.to_csv(Config.RESULT_LOG, index=False)
    print(f"📁 Metrics saved to {Config.RESULT_LOG}")
    
    return {
        "accuracy": acc,
        "precision": p_score,
        "recall": r,
        "f1": f1
    }

# === Save Predictions ===
def save_predictions(trainer, dataset):
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    texts = dataset["text"]  # Extract original texts

    df = pd.DataFrame({
        "original_text": texts,
        "true_label": true_labels,
        "predicted_label": preds
    })

    df.to_csv(Config.PREDICTION_LOG, index=False)
    print(f"📁 Predictions saved to {Config.PREDICTION_LOG}")

# === Main ===
def main():
    setup_environment()
    df = load_and_process_data()
    dataset, tokenizer = tokenize_and_prepare_dataset(df)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        num_train_epochs=Config.EPOCHS,
        learning_rate=Config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.EARLY_STOPPING_PATIENCE)]
    )

    print("🚀 Training MARBERT...")
    trainer.train()
    print("💾 Saving model...")
    trainer.save_model(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    print("📊 Final Evaluation:")
    trainer.evaluate()
    save_predictions(trainer, dataset["test"])

if __name__ == "__main__":
    main()
