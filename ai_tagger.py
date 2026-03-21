import os
import sys

# Add external libs to path
sys.path.append('/media/liza/B779-017B/ai/python_libs')

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import csv
from tqdm import tqdm

def ai_tag_all(raw_dir, output_csv):
    print("Loading CLIP AI Model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define our semantic search categories
    categories = {
        'length': ["short nails", "medium length nails", "long nails"],
        'color': ["nude nails", "dark nails", "bright nails", "pastel nails", "glitter nails"],
        'design': ["solid color nails", "french tip nails", "minimalistic nail art", "patterned nail art", "chrome nails", "matte nails"],
        'shape': ["oval nails", "square nails", "almond nails", "coffin nails", "stiletto nails"]
    }

    images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    for img_name in tqdm(images, desc="AI Tagging"):
        try:
            img_path = os.path.join(raw_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            
            tags = {'filename': img_name, 'liked': 'yes'}
            
            for cat_name, prompts in categories.items():
                inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                best_idx = probs.argmax().item()
                # Clean up the prompt to just the label (e.g., "short nails" -> "short")
                tag = prompts[best_idx].replace(" nails", "").replace(" nail art", "").replace(" color", "").replace(" length", "").replace(" tip", "")
                tags[cat_name] = tag
            
            results.append(tags)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'liked', 'length', 'color', 'design', 'shape'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nAI Tagging complete! Data saved to {output_csv}")

if __name__ == "__main__":
    ai_tag_all('raw_images', 'nail_data.csv')
