import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def extract_features(img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize for consistency
    img = cv2.resize(img, (128, 128))
    
    # 1. Color Histogram (Global color preference)
    # Convert to HSV as it's better for color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()
    
    # 2. HOG Features (Shape and texture)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False)
    
    # Combine features
    return np.hstack([color_features, hog_features])

def load_dataset(base_path):
    X = []
    y = []
    
    categories = {'yes': 1, 'no': 0}
    
    for label_name, label_val in categories.items():
        dir_path = os.path.join(base_path, label_name)
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            features = extract_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label_val)
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    dataset_path = 'dataset'
    print("Loading dataset and extracting features (HOG + Color)...")
    X, y = load_dataset(dataset_path)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features per sample.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train SVM
    print("Training SVM with RBF kernel...")
    clf = SVC(kernel='rbf', probability=True, C=1.0)
    
    # Requirement: Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    clf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = clf.predict(X_test)
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVM Confusion Matrix - Nail Preference')
    plt.savefig('svm_confusion_matrix.png')
    print("Confusion matrix saved to 'svm_confusion_matrix.png'")
