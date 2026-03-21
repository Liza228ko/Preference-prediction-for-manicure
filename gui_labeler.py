import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class LabelerApp:
    def __init__(self, master, raw_dir, yes_dir, no_dir):
        self.master = master
        self.raw_dir = raw_dir
        self.yes_dir = yes_dir
        self.no_dir = no_dir
        
        os.makedirs(self.yes_dir, exist_ok=True)
        os.makedirs(self.no_dir, exist_ok=True)
        
        self.images = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_idx = 0
        
        if not self.images:
            messagebox.showinfo("Done", "No more images in raw_images!")
            master.quit()
            return
            
        # UI Setup
        self.header = tk.Label(master, text="", font=('Helvetica', 12, 'bold'))
        self.header.pack(pady=5)
        
        self.warning_label = tk.Label(master, text="", font=('Helvetica', 10, 'bold'), fg="red")
        self.warning_label.pack()

        self.img_container = tk.Label(master, bg="gray90")
        self.img_container.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        self.btn_frame = tk.Frame(master)
        self.btn_frame.pack(pady=15)
        
        font = ('Helvetica', 12, 'bold')
        tk.Button(self.btn_frame, text="LIKE (y)", bg='#90EE90', font=font, command=self.mark_yes, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="NO (n)", bg='#FFB6C1', font=font, command=self.mark_no, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="Skip (s)", font=font, command=self.skip, width=8).pack(side=tk.LEFT, padx=5)
        tk.Button(self.btn_frame, text="DELETE (d)", bg='#D3D3D3', font=font, command=self.delete_img, width=8).pack(side=tk.LEFT, padx=5)
        
        master.bind('y', lambda e: self.mark_yes())
        master.bind('n', lambda e: self.mark_no())
        master.bind('s', lambda e: self.skip())
        master.bind('d', lambda e: self.delete_img())
        
        self.show_image()
        
    def show_image(self):
        if self.current_idx >= len(self.images):
            messagebox.showinfo("Done", "All images labeled!")
            self.master.quit()
            return
            
        img_path = os.path.join(self.raw_dir, self.images[self.current_idx])
        try:
            img = Image.open(img_path)
            w, h = img.size
            
            # Detect Collage (Wide aspect ratio)
            if w / h > 1.35:
                self.warning_label.config(text="⚠️ COLLAGE DETECTED (Too Wide)")
            else:
                self.warning_label.config(text="")
            
            # Resize image to fit comfortably
            img.thumbnail((700, 600))
            self.photo = ImageTk.PhotoImage(img)
            self.img_container.config(image=self.photo)
            
            self.header.config(text=f"Image {self.current_idx + 1} of {len(self.images)}: {self.images[self.current_idx]}")
            self.master.title(f"Nail Labeler - {self.images[self.current_idx]}")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            self.skip()
            
    def mark_yes(self):
        self.move_image(self.yes_dir)
        
    def mark_no(self):
        self.move_image(self.no_dir)
        
    def skip(self):
        self.current_idx += 1
        self.show_image()
        
    def delete_img(self):
        img_name = self.images[self.current_idx]
        src = os.path.join(self.raw_dir, img_name)
        try:
            os.remove(src)
            print(f"Deleted {img_name}")
        except Exception as e:
            print(f"Error deleting {src}: {e}")
        self.current_idx += 1
        self.show_image()

    def move_image(self, target_dir):
        img_name = self.images[self.current_idx]
        src = os.path.join(self.raw_dir, img_name)
        dst = os.path.join(target_dir, img_name)
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"Error moving {src}: {e}")
        self.current_idx += 1
        self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("850x850")
    # Center the window
    root.eval('tk::PlaceWindow . center')
    app = LabelerApp(root, 'raw_images', 'dataset/yes', 'dataset/no')
    root.mainloop()
