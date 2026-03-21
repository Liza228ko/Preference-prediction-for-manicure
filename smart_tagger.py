import csv
import os
import shutil

def smart_auto_label(data_csv, raw_dir, yes_dir, no_dir):
    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)
    
    # Define our 'Like' profile based on your labels:
    # Colors you liked: nude, pastel
    # Designs: french, minimalistic, solid
    # Length: short, medium
    # Shape: oval, almond, square
    
    # We'll assign a 'score' to each raw image based on these preferences
    
    with open(data_csv, 'r') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)
        
    labeled_count = 0
    
    for row in all_data:
        filename = row['filename']
        src_path = os.path.join(raw_dir, filename)
        
        # Only process if image is still in raw_images
        if not os.path.exists(src_path):
            continue
            
        score = 0
        # Scoring logic
        if row['color'] in ['nude', 'pastel']: score += 2
        if row['design'] in ['french', 'minimalistic', 'solid']: score += 2
        if row['length'] in ['short', 'medium']: score += 1
        if row['shape'] in ['oval', 'almond', 'square']: score += 1
        
        # Penalties for things you usually dislike (bright, chrome, stiletto, long)
        if row['color'] == 'bright': score -= 2
        if row['design'] == 'chrome': score -= 2
        if row['shape'] == 'stiletto': score -= 2
        if row['length'] == 'long': score -= 1
        
        # Decision threshold (e.g., score >= 3 is likely a 'Yes')
        if score >= 3:
            shutil.move(src_path, os.path.join(yes_dir, filename))
            labeled_count += 1
            print(f"Auto-Liked: {filename} (Score: {score})")
        elif score <= -2:
            # We can also auto-dislike the very obvious ones
            shutil.move(src_path, os.path.join(no_dir, filename))
            print(f"Auto-Disliked: {filename} (Score: {score})")

    print(f"\nSmart tagging complete! Automatically labeled {labeled_count} as 'Yes'.")

if __name__ == "__main__":
    # Note: We use nail_data.csv (AI guesses) as the base for our raw_images
    smart_auto_label('nail_data.csv', 'raw_images', 'dataset/yes', 'dataset/no')
