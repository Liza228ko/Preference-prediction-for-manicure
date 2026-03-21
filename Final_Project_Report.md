# Project #1: Personalized Aesthetic Preference Prediction for Manicure Designs
**Course:** Undergraduate AI Capstone (NYCU Spr2026)  
**Student Name:** Yelyzaveta Kozachenko  
**Student ID:** [YOUR_STUDENT_ID]  
**Dataset Link:** [YOUR_GITHUB_LINK_HERE]

---

## 1. Research Question and Motivation
**Research Question:**  
Can machine learning models effectively distinguish personal aesthetic preferences in manicure designs by comparing high-level semantic features (Method 1) against low-level pixel data (Method 2)?

**Motivation:**  
Aesthetic taste is highly subjective and difficult to quantify using general recommendation systems. For manicure designs, specific attributes like "nude colors," "minimalist art," or "almond shapes" often define a user's preference. By creating a personalized "Nail Preference" dataset, this project investigates the "semantic gap" in computer vision. It aims to prove that human-interpretable features provide a more robust basis for personal style prediction than raw pixels alone, especially when dealing with limited data (small-scale datasets).

---

## 2. Documentation of Dataset
- **Data Type:** RGB Images (.jpg, .png) and categorical metadata (.csv).
- **External Source:** Images were scraped from Bing Image Search using a custom Python crawler. Search queries included: "korean minimalist nails," "micro french manicure," "nude nail designs," and "pearl chrome nails."
- **Amount and Composition:**
  - Total Labeled Samples: 337
  - Liked (Yes): 133 images
  - Disliked (No): 204 images
- **Data Collection Process:**
  - **Scraping:** Automated collection via `downloader_v3.py`.
  - **AI-Assisted Tagging:** Every image was processed using the **OpenAI CLIP** (Vision-Language) model to generate "initial guesses" for four semantic categories: Length, Color, Design, and Shape.
  - **Human-in-the-Loop Labeling:** I developed a custom Tkinter-based "GUI Labeler" to manually confirm/correct the AI's guesses and provide the ground-truth "Yes/No" preference label.
- **Dataset Examples:**
  - *Liked:* Typically feature nude colors, oval shapes, and minimalistic designs.
  - *Disliked:* Often feature chrome textures, bright neon colors, or stiletto shapes.

---

## 3. Description of Methods
### Method 1: Linear SVM with Semantic Features (Classical ML)
- **Feature Engineering:** Categorical tags (e.g., "nude," "french") were converted into numerical data using **One-Hot Encoding**.
- **Algorithm:** Support Vector Machine (SVM) with a **Linear Kernel** (via Scikit-Learn).
- **Rationale:** The linear kernel allows for "Feature Importance" analysis, revealing which specific descriptors (like 'nude') drive the preference model.

### Method 2: Fine-Tuned ResNet-18 (Deep Learning)
- **Algorithm:** ResNet-18 (via PyTorch).
- **Transfer Learning:** Used weights pre-trained on ImageNet.
- **Fine-Tuning Strategy:** To improve upon basic feature extraction, I **unfroze the final residual block (layer4)** of the ResNet. This allowed the model to adapt its deeper visual filters to the specific textures and shapes of nail art.
- **Optimization:** Trained using SGD with momentum (0.9) and a learning rate of 0.001 for 12 epochs.

---

## 4. Description of Experiments and Results
### 4.1. Comparative Performance
| Metric | Method 1: SVM (Semantic) | Method 2: ResNet-18 (Pixels) |
| :--- | :---: | :---: |
| **Accuracy** | **96.30%** | **71.00%** |
| **Precision (Liked)** | 0.96 | 0.68 |
| **Recall (Liked)** | 1.00 | 0.59 |
| **F1-Score (Liked)** | 0.98 | 0.63 |

**Analysis:**  
The SVM achieved a near-perfect score, demonstrating that subjective taste is highly correlated with semantic descriptors. The ResNet-18 performed respectably (71%), significantly outperforming a baseline MobileNetV2 (57%) thanks to the fine-tuning of `layer4`.

### 4.2. Feature Importance (SVM Analysis)
The Linear SVM weights reveal the primary drivers of preference:
- **Top Positive Drivers:** `color_nude`, `color_pastel`, `design_minimalistic`.
- **Top Negative Drivers:** `design_chrome`, `color_glitter`, `color_bright`.

*(See Figure: `nail_preference_importance.png` in Appendix)*

### 4.3. Impact of Training Data Size
I evaluated how the SVM's accuracy changes as the amount of training data increases:
- **20% Data:** 66.67% accuracy
- **60% Data:** 87.50% accuracy
- **100% Data:** 96.30% accuracy
**Conclusion:** The model shows a strong "learning curve," suggesting that even a small increase in labeled data (from 200 to 300) significantly stabilized the aesthetic prediction.

### 4.4. Dimensionality Reduction (PCA)
I applied PCA to the one-hot encoded semantic features to reduce the feature space:
- **Full Features (Baseline):** 96.30% accuracy
- **PCA (Reduced to 5 components):** 92.59% accuracy
**Conclusion:** Reducing dimensionality slightly decreased accuracy, indicating that specific, rare tags (like 'coffin' shape or 'matte' design) carry unique information that PCA compresses away.

---

## 5. Discussion
- **Observed Behaviors:** The results confirm the "Semantic Gap." Deep learning models struggle to find the "concept" of beauty in raw pixels without massive data. However, by pre-tagging images with human-readable concepts (CLIP), the classification task becomes trivial for a simple SVM.
- **Factors Affecting Results:** The class imbalance (204 Disliked vs 133 Liked) initially caused the CNN to over-predict "No." Fine-tuning and data augmentation (flips/rotations) were essential to help the CNN identify "Yes" patterns.
- **Future Work:** If more time were available, I would implement **Siamese Networks** to learn a "similarity metric" between nail designs, allowing the system to recommend designs similar to those I've liked in the past.

---

## 6. List of References
1. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
2. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR (ResNet).
4. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP).

---

## Appendix: Figures and Code
### Figures
1. **`nail_preference_importance.png`**: Shows the SVM weights for each tag.
2. **`final_svm_cm.png`**: Confusion matrix for the semantic SVM.
3. **`resnet_confusion_matrix.png`**: Confusion matrix for the fine-tuned ResNet-18.

### Core Program Code
*(The full code is attached in the ZIP/Github, below are the primary training blocks)*

**SVM Training Block (`train_final_svm.py`):**
```python
df_encoded = pd.get_dummies(df[feature_cols])
model = SVC(kernel='linear', probability=True)
cv_scores = cross_val_score(model, X, y, cv=5)
model.fit(X_train, y_train)
```

**ResNet Fine-Tuning Block (`train_resnet.py`):**
```python
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters(): param.requires_grad = False
for param in model.layer4.parameters(): param.requires_grad = True # Fine-tune last block
model.fc = nn.Linear(model.fc.in_features, 2)
```
