import os
import cv2
import numpy as np
import csv

def guess_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w, _ = img.shape
    
    # 1. Guess Color Type (using average brightness and saturation)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_v = np.mean(hsv[:,:,2]) # Brightness
    avg_s = np.mean(hsv[:,:,1]) # Saturation
    
    if avg_s < 40: color = "nude"
    elif avg_v < 80: color = "dark"
    elif avg_s > 150: color = "bright"
    else: color = "pastel"
    
    # 2. Guess Length (using aspect ratio of the image as a proxy)
    # Most nail photos are portrait if they are long nails
    ratio = h / w
    if ratio > 1.2: length = "long"
    elif ratio > 1.0: length = "medium"
    else: length = "short"
    
    # 3. Default guesses for others (to be reviewed by you)
    design = "solid"
    shape = "oval"
    
    return {
        'filename': os.path.basename(img_path),
        'liked': 'yes', # Default
        'length': length,
        'color': color,
        'design': design,
        'shape': shape
    }

def auto_tag_all(raw_dir, output_csv):
    images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'liked', 'length', 'color', 'design', 'shape'])
        writer.writeheader()
        
        for i, img_name in enumerate(images):
            print(f"[{i+1}/{len(images)}] Processing {img_name}...")
            res = guess_features(os.path.join(raw_dir, img_name))
            if res:
                writer.writerow(res)

if __name__ == "__main__":
    auto_tag_all('raw_images', 'nail_data.csv')
