import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

def get_svm_auroc():
    all_rows = []
    master_df = pd.read_csv('nail_data.csv')
    for cat in ['yes', 'no']:
        cat_dir = os.path.join('dataset', cat)
        if not os.path.exists(cat_dir): continue
        for fname in os.listdir(cat_dir):
            row = master_df[master_df['filename'] == fname]
            if not row.empty:
                r_dict = row.iloc[0].to_dict()
                r_dict['liked'] = 1 if cat == 'yes' else 0
                all_rows.append(r_dict)
    df = pd.DataFrame(all_rows)
    X = pd.get_dummies(df[['length', 'color', 'design', 'shape']])
    y = df['liked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, probs)

if __name__ == "__main__":
    svm_auc = get_svm_auroc()
    print(f"SVM AUROC: {svm_auc:.4f}")
