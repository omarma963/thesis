# pip install emoji==0.6.0
from tqdm.notebook import tqdm
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np
import os
import logging
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ======================
# CONFIGURATION
# ======================
CONFIG = {
    "dataset_paths": {
        "en": "/content/drive/MyDrive/english_dataset_balanced.csv",
        "blog": "/content/drive/MyDrive/blog_dataset_balanced.csv"
    },
    "model_name": "vinai/bertweet-base",
    "output_dir": "./model_output",
    "text_columns": ["text", "content"],
    "label_columns": ["label", "gender"],
    "max_length": 128,
    "test_size": 0.1,
    "seed": 42,
    "num_labels": 2,
    "early_stopping_patience": 2,
    "learning_rate": 2e-5,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "weight_decay": 0.01,
    "num_train_epochs": 6,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 1,
    "fp16": True if torch.cuda.is_available() else False
}

# ======================
# SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ======================
# DATA PROCESSING
# ======================
class DataProcessor:
    @staticmethod
    def load_and_validate_dataset(path):
        try:
            df = pd.read_csv(path, on_bad_lines='warn')
            text_col = next((col for col in CONFIG["text_columns"] if col in df.columns), None)
            if not text_col:
                raise ValueError(f"No valid text column found in {path}")
            label_col = next((col for col in CONFIG["label_columns"] if col in df.columns), None)
            if not label_col:
                raise ValueError(f"No valid label column found in {path}")
            df = df.rename(columns={text_col: "text", label_col: "label"})
            
            # More thorough data validation
            if df.empty:
                raise ValueError("Empty dataset")
            if df['text'].isna().any():
                logger.warning(f"Found {df['text'].isna().sum()} NaN values in text - dropping them")
                df = df.dropna(subset=['text'])
            if df['label'].isna().any():
                logger.warning(f"Found {df['label'].isna().sum()} NaN values in labels - dropping them")
                df = df.dropna(subset=['label'])
                
            return df[['text', 'label']]
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
            raise

    @staticmethod
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = (
            text.replace('\x00', '')
            .replace('\ufffd', '')
            .encode('ascii', 'ignore')
            .decode('ascii', 'ignore')
            .strip()
        )
        return text if text else "[EMPTY]"

    @staticmethod
    def preprocess_datasets():
        try:
            dfs = []
            for name, path in CONFIG["dataset_paths"].items():
                df = DataProcessor.load_and_validate_dataset(path)
                df['source'] = name
                dfs.append(df)

            combined = pd.concat(dfs, ignore_index=True)
            combined['text'] = combined['text'].apply(DataProcessor.clean_text)
            combined = combined[combined['text'] != "[EMPTY]"]
            
            # More robust label encoding
            combined['label'] = combined['label'].astype(str).str.lower()
            label_mapping = {'male': 0, 'female': 1, 'm': 0, 'f': 1, '0': 0, '1': 1}
            combined['label'] = combined['label'].map(label_mapping)
            
            # Handle any remaining unmapped labels
            if combined['label'].isna().any():
                logger.warning(f"Found {combined['label'].isna().sum()} unmapped labels - dropping them")
                combined = combined.dropna(subset=['label'])
            
            combined['label'] = combined['label'].astype(int)
            combined = combined.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)

            # Enhanced Class Distribution Report
            label_counts = combined['label'].value_counts().sort_index()
            label_mapping = {0: "Male", 1: "Female"}
            report = "\n".join(f"{label_mapping.get(k, k)}: {v} ({v/len(combined):.1%})" for k, v in label_counts.items())
            logger.info("📊 Class Distribution:\n" + report)
            
            # Save distribution plot
            plt.figure(figsize=(8, 6))
            sns.countplot(x='label', data=combined)
            plt.title('Class Distribution')
            plt.savefig(os.path.join(CONFIG["output_dir"], "class_distribution.png"))
            plt.close()
            
            dist_file = os.path.join(CONFIG["output_dir"], "class_distribution.txt")
            with open(dist_file, "w", encoding="utf-8") as f:
                f.write("Class Distribution Report\n\n")
                f.write(report)
                f.write(f"\n\nTotal samples: {len(combined)}")
            logger.info(f"📄 Class distribution saved to {dist_file}")

            return Dataset.from_pandas(combined)

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

# ======================
# TOKENIZATION
# ======================
class TokenizerWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"], use_fast=False, normalization=True
        )
        self.vocab_size = len(self.tokenizer)
        logger.info(f"Tokenizer initialized with vocab size: {self.vocab_size}")

    def tokenize_with_validation(self, batch):
        try:
            tokenized = self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=CONFIG["max_length"],
                return_tensors="pt"
            )
            if (tokenized["input_ids"] >= self.vocab_size).any():
                raise ValueError("Token ID out of range")
            return tokenized
        except Exception as e:
            logger.warning(f"Tokenization failed, retrying with cleaned text: {str(e)}")
            cleaned = [DataProcessor.clean_text(t) for t in batch["text"]]
            return self.tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=CONFIG["max_length"],
                return_tensors="pt"
            )

# ======================
# MODEL TRAINING
# ======================
class ModelTrainer:
    @staticmethod
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        acc = accuracy_score(labels, preds)
        
        # Additional metrics
        cm = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=["Male", "Female"], output_dict=True)
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Male", "Female"], yticklabels=["Male", "Female"])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"))
        plt.close()
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'male_precision': report['Male']['precision'],
            'male_recall': report['Male']['recall'],
            'female_precision': report['Female']['precision'],
            'female_recall': report['Female']['recall']
        }

    @staticmethod
    def train():
        try:
            dataset = DataProcessor.preprocess_datasets()
            tokenizer = TokenizerWrapper()

            tokenized = dataset.map(
                tokenizer.tokenize_with_validation,
                batched=True,
                batch_size=1000,
                remove_columns=['text', 'source']
            )

            split = tokenized.train_test_split(test_size=CONFIG["test_size"], seed=CONFIG["seed"])

            model = AutoModelForSequenceClassification.from_pretrained(
                CONFIG["model_name"], 
                num_labels=CONFIG["num_labels"],
                ignore_mismatched_sizes=True
            )

            # Enhanced training arguments
            args = TrainingArguments(
                output_dir=CONFIG["output_dir"],
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=CONFIG["learning_rate"],
                per_device_train_batch_size=CONFIG["train_batch_size"],
                per_device_eval_batch_size=CONFIG["eval_batch_size"],
                num_train_epochs=CONFIG["num_train_epochs"],
                weight_decay=CONFIG["weight_decay"],
                warmup_ratio=CONFIG["warmup_ratio"],
                gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
                fp16=CONFIG["fp16"],
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                logging_dir="./logs",
                logging_steps=100,
                report_to="none",
                save_total_limit=2,
                seed=CONFIG["seed"]
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=split["train"],
                eval_dataset=split["test"],
                processing_class=tokenizer.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer.tokenizer),
                compute_metrics=ModelTrainer.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["early_stopping_patience"])]
            )

            logger.info("🚀 Starting training...")
            train_result = trainer.train()
            
            # Save training metrics
            metrics = train_result.metrics
            trainer.save_metrics("train", metrics)
            logger.info(f"📊 Training metrics: {metrics}")

            logger.info("✅ Training complete.")

            logger.info("📊 Evaluating...")
            results = trainer.evaluate()
            logger.info(f"📈 Evaluation results: {results}")

            # Save full evaluation report
            eval_report = {
                "config": CONFIG,
                "metrics": results,
                "timestamp": datetime.now().isoformat()
            }
            eval_file = os.path.join(CONFIG["output_dir"], "evaluation_report.json")
            with open(eval_file, "w") as f:
                json.dump(eval_report, f, indent=2)
            logger.info(f"📄 Evaluation report saved to {eval_file}")

            trainer.save_model(CONFIG["output_dir"])
            tokenizer.tokenizer.save_pretrained(CONFIG["output_dir"])

            return trainer, results

        except Exception as e:
            logger.critical(f"Training failed: {str(e)}")
            raise

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    try:
        trainer, eval_results = ModelTrainer.train()
        logger.info("✅ Finished all stages successfully.")
    except Exception as e:
        logger.error(f"🔥 Unhandled error: {str(e)}")