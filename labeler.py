import os
import shutil
import subprocess

def label_images(raw_dir='raw_images', yes_dir='dataset/yes', no_dir='dataset/no'):
    # Ensure directories exist
    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)

    images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print("No images found in raw_images. Run downloader.py first!")
        return

    print(f"Found {len(images)} images to label.")
    print("Commands: 'y' (Like), 'n' (Dislike), 's' (Skip), 'q' (Quit)")

    for img_name in images:
        img_path = os.path.join(raw_dir, img_name)
        
        # Open image with system viewer (Linux)
        viewer = subprocess.Popen(['xdg-open', img_path])
        
        choice = input(f"Label '{img_name}': ").lower().strip()
        
        # Close viewer (optional, but cleaner)
        viewer.terminate()

        if choice == 'y':
            shutil.move(img_path, os.path.join(yes_dir, img_name))
            print(f"Moved to 'yes' folder.")
        elif choice == 'n':
            shutil.move(img_path, os.path.join(no_dir, img_name))
            print(f"Moved to 'no' folder.")
        elif choice == 's':
            print(f"Skipped.")
            continue
        elif choice == 'q':
            print("Stopping labeling.")
            break
        else:
            print("Invalid input, skipping.")

if __name__ == "__main__":
    label_images()
