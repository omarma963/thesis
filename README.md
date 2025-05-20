# Detect male users pretending to be female on Facebook by analyzing writing style (Arabic + English), 
# using deep learning (BERT/AraBERT), and image verification.

Overview

This project investigates the detection of male users impersonating female users on Facebook using linguistic patterns and profile image verification. The system is built using transformer-based models (BERT and AraBERT) and integrates reverse image search via SerpAPI.

Contents

main.py: Main script for training and evaluation

data/: Contains training and evaluation datasets

notebooks/: Jupyter notebooks for exploratory data analysis

models/: Fine-tuned model checkpoints

utils/: Utility scripts for preprocessing and image verification

Setup

Install dependencies:

pip install -r requirements.txt

Set up SerpAPI key:

export SERPAPI_API_KEY='your_key_here'

Run training:

python main.py --train

Dataset Sources

Arabic: PAN 2017 Author Profiling

English: Blog Authorship Corpus

Citation

If using this work, please cite the thesis:

[Your Name], "Detecting Girl Impersonators on Social Media Through Writing Style/Patterns Using Deep Learning," Thesis, [Your University], 2025.
