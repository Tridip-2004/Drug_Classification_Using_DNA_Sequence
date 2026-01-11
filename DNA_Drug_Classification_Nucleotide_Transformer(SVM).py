"""
DNA Sequence-Based Drug Classification using SVM
------------------------------------------------
Paper-ready Python script

Author: Tridip Panja
Description:
This script implements a machine learning-based approach for
drug classification using DNA nucleotide sequences and SVM.
"""

# =========================
# 1. Import Libraries
# =========================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 2. Dataset Loading
# =========================
def load_dataset(path):
    """
    Load DNA sequence dataset
    Expected format:
    - column 0: DNA sequence
    - column 1: class label
    """
    data = pd.read_csv(path)
    sequences = data.iloc[:, 0].values
    labels = data.iloc[:, 1].values
    return sequences, labels


# =========================
# 3. Feature Extraction
# =========================
def dna_to_kmer_features(sequences, k=3):
    """
    Convert DNA sequences into k-mer frequency features
    """
    from collections import Counter

    features = []
    for seq in sequences:
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        counts = Counter(kmers)
        features.append(counts)

    return pd.DataFrame(features).fillna(0)


# =========================
# 4. Model Training
# =========================
def train_svm(X, y):
    """
    Train Support Vector Machine classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model


# =========================
# 5. Main Function
# =========================
if __name__ == "__main__":
    # Example usage (update dataset path)
    dataset_path = "dna_drug_dataset.csv"

    sequences, labels = load_dataset(dataset_path)
    X = dna_to_kmer_features(sequences, k=3)

    model = train_svm(X, labels)
