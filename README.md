# 🧬 Drug Classification Using DNA Sequences
📌 Internship Project Summary

This repository contains the complete work completed during my first internship project on Drug Classification using DNA Sequences at JISIASR, where I explored classical machine learning models combined with state-of-the-art pretrained DNA language models to classify drugs based on genomic information.

The primary goal of this project was to analyze how pretrained nucleotide models can be leveraged for downstream drug classification tasks, and to compare their performance with different machine learning classifiers.

# 🎯 Project Objectives

Understand and preprocess DNA sequence data for machine learning

Extract meaningful embeddings from pretrained DNA models

Experiment with multiple machine learning classifiers

Evaluate model performance and identify the best-performing approach

Gain hands-on experience with bioinformatics + AI

# 🧪 Models & Techniques Used
🔹 Pretrained DNA Models

DNABERT

Nucleotide Transformer (NT)

These models were used to extract sequence-level embeddings from raw DNA sequences.

# 🔹 Machine Learning Models

Support Vector Machine (SVM)

Random Forest

XGBoost ⭐

# 🏆 Best Results
Model Combination	Accuracy
Nucleotide Transformer + XGBoost	74% ✅
DNABERT + XGBoost	Competitive
Other ML models	Lower accuracy

📌 Highest accuracy achieved: 74% using XGBoost with Nucleotide Transformer embeddings

# ⚙️ Methodology

DNA Sequence Preprocessing

Cleaning and formatting raw DNA sequences

Tokenization suitable for pretrained DNA models

Feature Extraction

Generated embeddings using:

DNABERT

Nucleotide Transformer

Model Training

Trained multiple ML classifiers on extracted embeddings

Hyperparameter tuning for performance optimization

Evaluation

Accuracy as the primary metric

Comparative analysis across models

# 🧬 Tech Stack

Programming Language: Python

Libraries & Frameworks:

PyTorch

Hugging Face Transformers

Scikit-learn

XGBoost

NumPy

Pandas

# 📂 Project Structure
├── data/
│   ├── raw_sequences/
│   └── processed_data/
│
├── models/
│   ├── dnabert/
│   └── nucleotide_transformer/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_extraction.ipynb
│   └── model_training.ipynb
│
├── results/
│   └── performance_metrics/
│
├── requirements.txt
└── README.md

# 🚀 Key Learnings

Practical understanding of genomic data representation

Hands-on experience with pretrained DNA language models

Comparative analysis of ML models on biological embeddings

Importance of feature quality over model complexity

# 🔮 Future Work

Fine-tuning DNABERT and NT models

Exploring deep learning classifiers

Increasing dataset size for better generalization

Applying explainability techniques (e.g., SHAP)

# 👨‍💻 Author

Tridip Panja
Intern – Drug Classification Using DNA Sequences

# ⭐ Acknowledgements

I would like to thank my internship mentors and organization for providing the opportunity to work on this interdisciplinary project combining bioinformatics and machine learning
