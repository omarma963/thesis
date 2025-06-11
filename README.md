# Bilingual Deep Learning Framework for Identifying Gender Impersonation on Social Media

## 🧠 Overview
This repository contains the full implementation of the thesis project:  
**"A Bilingual Deep Learning Framework for Identifying Gender Impersonation on Social Media via Text, Image, and Metadata Fusion"**  
by *E. Mohammad Omar Mahairi*  
Supervised by *Dr. Basel Alkhateb* (Director, SVU Web Science Department)

## 🗃 Datasets
- **PAN 2017 (Arabic + English)**
- **Blog Authorship Corpus (English)**
- Cleaned and balanced subsets are included under `data/`.

## 🔍 Models
- **Arabic**: MARBERT fine-tuned on PAN 2017 Arabic
- **English**: BERTweet trained on PAN 2017 English + Blog Corpus
- Saved under `models/`

## 🔄 Pipeline Steps
1. Text-based gender classification (Arabic/English)
2. Instagram link detection (via regex & scraping)
3. Image authenticity (Google Lens via SerpAPI)
4. EXIF metadata scoring
5. Final Decision Layer (weighted scoring or meta-classifier)

## 🧩 Decision Layer
Found under `decision_layer/`. Combines:
- `text_score`, `image_score`, `exif_score`, `social_score`
- Custom weights: 30% text, 40% image, 15% EXIF, 15% social
- Supports fallback to threshold or logistic regression meta-classifier

## 🏃‍♂️ Training Scripts
Found in `training/`, includes:
- `train_arabic_marbert.py`
- `train_english_bertweet.py`
- Early stopping, validation, logging, and model saving

## 📊 Results
- **Arabic Model (MARBERT)**: F1 = 0.7862, Accuracy = 0.7666
- **English Model (BERTweet)**: F1 = 0.7202, Accuracy = 0.7118

## 📷 Instagram & Image Features
- `instagram_detector.py`: checks for IG links in profile text or HTML
- `image_exif_serpapi.py`: reverse image search + EXIF scoring

## 📘 Thesis
PDF version of the thesis in `thesis/final_thesis.pdf`  
Includes full methodology, experiments, and visual workflow.

## 📦 Setup
```bash
pip install -r requirements.txt

## Models and Datasets
English Bert Model ----> 
