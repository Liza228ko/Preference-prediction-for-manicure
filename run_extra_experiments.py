import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def load_data():
    all_rows = []
    categories = ['yes', 'no']
    master_df = pd.read_csv('nail_data.csv')
    for cat in categories:
        cat_dir = os.path.join('dataset', cat)
        if not os.path.exists(cat_dir): continue
        for fname in os.listdir(cat_dir):
            row = master_df[master_df['filename'] == fname]
            if not row.empty:
                r_dict = row.iloc[0].to_dict()
                r_dict['liked'] = 1 if cat == 'yes' else 0
                all_rows.append(r_dict)
    df = pd.DataFrame(all_rows)
    feature_cols = ['length', 'color', 'design', 'shape']
    X = pd.get_dummies(df[feature_cols])
    y = df['liked']
    return X, y

def run_pca_experiment():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline (No PCA)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, model.predict(X_test))
    
    # With PCA (Reduce to 5 components)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    model_pca = SVC(kernel='linear')
    model_pca.fit(X_p_train, y_p_train)
    pca_acc = accuracy_score(y_p_test, model_pca.predict(X_p_test))
    
    print(f"PCA EXPERIMENT:")
    print(f"Baseline (Full Features): {base_acc:.4f}")
    print(f"With PCA (5 components): {pca_acc:.4f}")

def run_data_size_experiment():
    X, y = load_data()
    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    print("\nDATA SIZE EXPERIMENT (SVM):")
    for size in sizes:
        if size < 1.0:
            X_sub, _, y_sub, _ = train_test_split(X, y, train_size=size, random_state=42)
        else:
            X_sub, y_sub = X, y
            
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Training Size {int(size*100)}%: Accuracy {acc:.4f}")

if __name__ == "__main__":
    run_pca_experiment()
    run_data_size_experiment()
