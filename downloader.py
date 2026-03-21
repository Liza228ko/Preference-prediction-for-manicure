import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO

def download_images(query, max_images=100, save_dir='raw_images'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Searching for '{query}'...")
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        
        count = 0
        for i, result in enumerate(results):
            if count >= max_images:
                break
                
            image_url = result['image']
            try:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content))
                
                # Basic cleaning: skip small images
                if image.size[0] < 200 or image.size[1] < 200:
                    continue
                
                # Convert to RGB if necessary (e.g., for PNG with alpha)
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                
                filename = f"{query.replace(' ', '_')}_{count}.jpg"
                image.save(os.path.join(save_dir, filename), "JPEG")
                print(f"Saved: {filename}")
                count += 1
            except Exception as e:
                print(f"Failed to download {image_url}: {e}")

if __name__ == "__main__":
    import sys
    search_query = "nail designs"
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
    
    download_images(search_query, max_images=50)
