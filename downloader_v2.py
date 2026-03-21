from bing_image_downloader import downloader
import os
import sys

def download_images(query, limit=50, output_dir='raw_images'):
    # Bing downloader creates a subfolder by default. 
    # We want everything in 'raw_images' for our labeler.
    temp_dir = 'temp_download'
    
    print(f"Downloading {limit} images for '{query}'...")
    downloader.download(query, limit=limit,  output_dir=temp_dir, 
                        adult_filter_off=True, force_replace=False, timeout=60, verbose=False)
    
    # Move files from temp subfolder to raw_images
    subfolder = os.path.join(temp_dir, query)
    if os.path.exists(subfolder):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in os.listdir(subfolder):
            src = os.path.join(subfolder, filename)
            # Create a unique name to avoid overwriting during multiple searches
            dst_filename = f"{query.replace(' ', '_')}_{filename}"
            dst = os.path.join(output_dir, dst_filename)
            os.rename(src, dst)
        
        # Cleanup
        os.rmdir(subfolder)
        os.rmdir(temp_dir)
        print(f"Success! Images moved to {output_dir}")
    else:
        print("Download failed or no images found.")

if __name__ == "__main__":
    search_query = "nail designs"
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:])
    
    download_images(search_query, limit=50)
