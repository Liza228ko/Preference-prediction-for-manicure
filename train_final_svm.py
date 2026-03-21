import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_preprocess():
    # Load all labeled data from the dataset folders
    all_rows = []
    categories = ['yes', 'no']
    
    # Load the features from our AI-tagged master list
    if not os.path.exists('nail_data.csv'):
        print("Error: nail_data.csv not found!")
        return None, None
        
    master_df = pd.read_csv('nail_data.csv')
    
    for cat in categories:
        cat_dir = os.path.join('dataset', cat)
        if not os.path.exists(cat_dir): continue
        
        for fname in os.listdir(cat_dir):
            row = master_df[master_df['filename'] == fname]
            if not row.empty:
                r_dict = row.iloc[0].to_dict()
                r_dict['liked'] = cat
                all_rows.append(r_dict)

    df = pd.DataFrame(all_rows)
    print(f"Total labeled samples for training: {len(df)}")
    
    # Encode categorical features into numbers (One-Hot Encoding for better SVM interpretation)
    feature_cols = ['length', 'color', 'design', 'shape']
    df_encoded = pd.get_dummies(df[feature_cols])
    
    df_encoded['liked'] = df['liked'].map({'yes': 1, 'no': 0})
    
    return df_encoded, df_encoded.columns.drop('liked')

if __name__ == "__main__":
    df, feature_names = load_and_preprocess()
    
    X = df[feature_names]
    y = df['liked']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM with Linear kernel to extract feature importance (weights)
    print("\nTraining Linear SVM for Feature Analysis...")
    model = SVC(kernel='linear', probability=True)
    
    # 5-Fold Cross Validation (Requirement)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f}")
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print("\nFinal Test Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    # FEATURE IMPORTANCE ANALYSIS
    # Get weights from the linear SVM
    weights = model.coef_[0]
    feat_importances = pd.Series(weights, index=feature_names)
    
    plt.figure(figsize=(10, 8))
    feat_importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Feature Importance: What drives your Nail Preference?')
    plt.xlabel('Importance (SVM Weight)')
    plt.tight_layout()
    plt.savefig('nail_preference_importance.png')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title('SVM Confusion Matrix')
    plt.savefig('final_svm_cm.png')
    
    print("\nAnalysis Complete!")
    print("- Confusion matrix saved to 'final_svm_cm.png'")
    print("- Feature importance saved to 'nail_preference_importance.png'")
