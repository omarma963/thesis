import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import os

class EnglishGenderPredictor:
    def __init__(self, model_path="path_to_your_finetuned_bert_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"✅ Model loaded on {self.device}")

    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        return text.strip()

    def predict(self, texts, batch_size=32, confidence_threshold=0.5):
        """Batch prediction with explicit 'uncertain' classification"""
        predictions = []
        confidences = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting genders"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            
            for prob in probs:
                male_prob, female_prob = prob
                max_prob = max(male_prob, female_prob)
                
                if max_prob < confidence_threshold:
                    gender = "uncertain"
                    confidence = 0.5  # Explicitly set confidence for uncertain cases
                else:
                    gender = "male" if male_prob > female_prob else "female"
                    confidence = max_prob  # Standard confidence score
                
                predictions.append(gender)
                confidences.append(confidence)

        return predictions, confidences

    def analyze_results(self, df):
        """Generate analysis for predictions"""
        print("\n📊 Class Distribution:")
        print(df["predicted_gender"].value_counts())
        
        confident_df = df[df["predicted_gender"] != "uncertain"]
        print("\n🔍 Confidence Analysis (for confident predictions):")
        print(confident_df["confidence"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))
        
        plt.figure(figsize=(10, 6))
        plt.hist(confident_df["confidence"], bins=20, range=(0, 1), color='#4e79a7', edgecolor='black')
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig("confidence_distribution.png")
        print("\n📈 Saved confidence distribution plot to confidence_distribution.png")

def main():
    INPUT_FILE = "english_data.txt"
    OUTPUT_FILE = "english_labeled_enhanced.csv"
    
    try:
        predictor = EnglishGenderPredictor()
        
        print("\n⏳ Loading data...")
        df = pd.read_csv(INPUT_FILE, header=None, names=["text"])
        df["text"] = df["text"].apply(predictor.clean_text)
        df = df[df["text"].str.len() > 0]
        print(f"✅ Loaded {len(df)} valid English texts")
        
        texts = df["text"].tolist()
        genders, confidences = predictor.predict(texts, confidence_threshold=0.5)
        
        df["predicted_gender"] = genders
        df["confidence"] = confidences
        df["male_prob"] = [1 - c if g == "female" else c for g, c in zip(genders, confidences)]
        df["female_prob"] = [c if g == "female" else 1 - c for g, c in zip(genders, confidences)]
        
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\n💾 Results saved to {OUTPUT_FILE}")
        
        predictor.analyze_results(df)
        
        print("\n🔎 Sample Predictions:")
        print(df[["text", "predicted_gender", "confidence"]].head(10))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
