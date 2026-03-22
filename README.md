# NailSense: Personalized Aesthetic Preference Dataset

This repository contains a custom-built dataset and classification models for predicting personal aesthetic preferences in manicure designs. This project was developed for the **Artificial Intelligence** course (NYCU Spr2026).

## 1. Dataset Documentation

### Data Type
The dataset consists of **RGB Images** in `.jpg`, `.jpeg`, and `.png` formats, accompanied by a master metadata file **`nail_data.csv`** containing semantic features and preference labels.

### External Source
Images were collected from **Bing Image Search** using a custom Python-based web crawler (`downloader_v3.py`). Search queries included:
* "korean minimalist nails"
* "micro french manicure"
* "nude nail designs"
* "pearl chrome nails"
* "dainty nail art"

### Amount and Composition
* **Total Labeled Samples:** 337
* **Liked (Yes):** 133 images
* **Disliked (No):** 204 images
* **Semantic Features:** 4 categories (Length, Color, Design, Shape)

### Dataset Structure
The project is organized as follows:
* `nail_data.csv`: Master metadata file.
* `dataset/yes/`: Images marked as "Liked".
* `dataset/no/`: Images marked as "Disliked".
* `gui_labeler_screenshot.png`: Screenshot of the custom labeling tool.

### Process of Data Collection
1. **Automated Scraping:** Images were downloaded into a `raw_images/` directory.
2. **AI-Assisted Tagging:** Every image was processed using the **OpenAI CLIP** (Vision-Language) model (`ai_tagger.py`) to generate initial guesses for semantic categories (e.g., "short", "nude", "minimalist").
3. **Human-in-the-Loop Labeling:** A custom **Tkinter-based GUI Tool** (`gui_labeler.py`) was developed to manually verify AI tags and provide the final ground-truth "Yes/No" preference labels.

### Examples
| Liked Example (Yes) | Disliked Example (No) |
| :---: | :---: |
| ![Yes](dataset/yes/korean_minimalist_nails_30_12.jpg) | ![No](dataset/no/manicure_designs_29.jpg) |
| *Nude, Minimalist, Oval* | *Bright, Patterned, Coffin* |

---

## 2. Classification Models

The project experiments with two primary methodologies to bridge the "semantic gap" in aesthetic prediction:

1. **Method 1: Linear SVM (Classical ML)**
   * Uses one-hot encoded semantic features from the CSV.
   * **Accuracy:** 96.30%
   * **AUROC:** 0.8864
2. **Method 2: Fine-Tuned ResNet-18 (Deep Learning)**
   * Uses raw pixel data via transfer learning.
   * **Accuracy:** 71.00%
   * **AUROC:** 0.7820

---

## 3. How to Run

### Requirements
* Python 3.10+
* PyTorch & Torchvision
* Scikit-Learn
* Pandas & Numpy
* PIL (Pillow)

### Training
To train the SVM model and see feature importance:
```bash
python3 train_final_svm.py
```

To train the ResNet-18 model:
```bash
python3 train_resnet.py
```

To run extra experiments (PCA, Data Size):
```bash
python3 run_extra_experiments.py
```

---

## 4. Student Information
* **Name:** Yelyzaveta Kozachenko
* **Student ID:** 111550205
