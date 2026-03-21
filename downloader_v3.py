import os
import requests
from bs4 import BeautifulSoup
import re
import json
from PIL import Image
from io import BytesIO

def download_bing_images(query, limit=50, output_dir='raw_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Searching Bing for: {query}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Bing search URL for images
    url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&form=HDRSC2&first=1"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all image result containers
        image_elements = soup.find_all('a', class_='iusc')
        
        count = 0
        for i, el in enumerate(image_elements):
            if count >= limit:
                break
                
            try:
                # Extract the image URL from the JSON data in the 'm' attribute
                m_attr = el.get('m')
                if not m_attr:
                    continue
                    
                m_data = json.loads(m_attr)
                img_url = m_data.get('murl')
                
                if not img_url:
                    continue

                # Download the image
                img_response = requests.get(img_url, headers=headers, timeout=5)
                img = Image.open(BytesIO(img_response.content))
                
                # Basic validation
                if img.size[0] < 150 or img.size[1] < 150:
                    continue
                
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Save
                ext = img_url.split('.')[-1].split('?')[0].lower()
                if ext not in ['jpg', 'jpeg', 'png']:
                    ext = 'jpg'
                
                filename = f"{query.replace(' ', '_')}_{count}.{ext}"
                save_path = os.path.join(output_dir, filename)
                img.save(save_path)
                
                print(f"[{count+1}/{limit}] Saved: {filename}")
                count += 1
                
            except Exception as e:
                # Silently skip individual failures
                continue

        print(f"Finished! Successfully downloaded {count} images.")
        
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    import sys
    query = "nail designs"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    
    download_bing_images(query, limit=50)
