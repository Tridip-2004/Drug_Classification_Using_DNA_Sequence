#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from imblearn.over_sampling import ADASYN
from sklearn.svm import SVC


# In[12]:


df = pd.read_excel('/content/Output.xlsx')


# In[13]:


df


# ## USING NT

# In[15]:


model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[16]:


df = df[['Sequence', 'Label']].dropna()
df['Label'] = df['Label'].astype(int)


# In[17]:


# get embeddings
def get_embedding(seq):
    tokens = tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

embeddings, labels = [], []

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        emb = get_embedding(row["Sequence"])
        embeddings.append(emb)
        labels.append(row["Label"])
    except Exception as e:
        print(f"⚠️ Error at index {i}: {e}")

X = np.array(embeddings)
y = np.array(labels)


# In[18]:


adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
print("✅ Dataset balanced using ADASYN.")


# In[19]:


# === 5-Fold Stratified Cross-Validation with XGBoost ===
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, precs, recalls, f1s, aucs = [], [], [], [], []

for fold, (train_index, test_index) in enumerate(kf.split(X_resampled, y_resampled), 1):
    X_train, X_test = X_resampled[train_index], X_resampled[test_index]
    y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    accs.append(accuracy_score(y_test, y_pred))
    precs.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recalls.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1s.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))


# In[20]:


if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_proba[:, 1])
else:
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
aucs.append(auc)

print(f"✅ Fold {fold} done.")


# In[21]:


print("\n=== Cross-Validation Results (5-Fold, XGBoost, ADASYN) ===")
print(f"Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"ROC AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


# In[ ]:





# In[ ]:




