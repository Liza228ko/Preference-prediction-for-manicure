import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import csv

class ReviewLabeler:
    def __init__(self, master, csv_path, raw_dir):
        self.master = master
        self.csv_path = csv_path
        self.raw_dir = raw_dir
        
        # Load all data from the auto-tagger
        self.data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        
        self.current_idx = 0
        self.results = []
        
        if not self.data:
            messagebox.showinfo("Done", "No data to review.")
            master.quit()
            return

        self.setup_ui()
        self.show_image()

    def setup_ui(self):
        # Image Display
        self.img_label = tk.Label(self.master, bg="gray80")
        self.img_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Control Panel
        panel = tk.Frame(self.master, width=300)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)

        tk.Label(panel, text="REVIEW FEATURES", font=('Helvetica', 16, 'bold')).pack(pady=10)
        tk.Label(panel, text="Correct if wrong, then press key", font=('Helvetica', 10, 'italic')).pack()

        # Feature Dropdowns (linked to vars)
        self.length_var = tk.StringVar()
        self.color_var = tk.StringVar()
        self.design_var = tk.StringVar()
        self.shape_var = tk.StringVar()

        tk.Label(panel, text="\nLength:").pack(anchor='w')
        ttk.Combobox(panel, textvariable=self.length_var, values=["short", "medium", "long"]).pack(fill='x')

        tk.Label(panel, text="\nColor:").pack(anchor='w')
        ttk.Combobox(panel, textvariable=self.color_var, values=["nude", "dark", "bright", "pastel", "glitter"]).pack(fill='x')

        tk.Label(panel, text="\nDesign:").pack(anchor='w')
        ttk.Combobox(panel, textvariable=self.design_var, values=["solid", "french", "minimalistic", "patterned", "chrome", "matte"]).pack(fill='x')

        tk.Label(panel, text="\nShape:").pack(anchor='w')
        ttk.Combobox(panel, textvariable=self.shape_var, values=["oval", "square", "almond", "coffin", "stiletto"]).pack(fill='x')

        # Legend
        tk.Label(panel, text="\n\nHOTKEYS:", font=('bold')).pack(anchor='w')
        tk.Label(panel, text="'y' = LIKE", fg="green").pack(anchor='w')
        tk.Label(panel, text="'n' = NO", fg="red").pack(anchor='w')
        tk.Label(panel, text="'s' = SKIP").pack(anchor='w')
        tk.Label(panel, text="'d' = DELETE").pack(anchor='w')

        # Bindings
        self.master.bind('y', lambda e: self.confirm_and_next("yes"))
        self.master.bind('n', lambda e: self.confirm_and_next("no"))
        self.master.bind('s', lambda e: self.skip())
        self.master.bind('d', lambda e: self.delete_img())

    def show_image(self):
        if self.current_idx >= len(self.data):
            self.finalize()
            return
            
        row = self.data[self.current_idx]
        img_path = os.path.join(self.raw_dir, row['filename'])
        
        if not os.path.exists(img_path):
            self.skip()
            return

        try:
            img = Image.open(img_path)
            img.thumbnail((750, 750))
            self.photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo)
            
            # Pre-fill with AI guesses
            self.length_var.set(row['length'])
            self.color_var.set(row['color'])
            self.design_var.set(row['design'])
            self.shape_var.set(row['shape'])
            
            self.master.title(f"Reviewing {self.current_idx+1}/{len(self.data)} - {row['filename']}")
        except:
            self.skip()

    def confirm_and_next(self, liked):
        row = self.data[self.current_idx]
        row.update({
            'liked': liked,
            'length': self.length_var.get(),
            'color': self.color_var.get(),
            'design': self.design_var.get(),
            'shape': self.shape_var.get()
        })
        self.results.append(row)
        
        # Move image to signal it's done
        dest_dir = f"dataset/{liked}"
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(os.path.join(self.raw_dir, row['filename']), os.path.join(dest_dir, row['filename']))
        
        self.current_idx += 1
        self.show_image()

    def skip(self):
        self.current_idx += 1
        self.show_image()

    def delete_img(self):
        img_path = os.path.join(self.raw_dir, self.data[self.current_idx]['filename'])
        if os.path.exists(img_path):
            os.remove(img_path)
        self.current_idx += 1
        self.show_image()

    def finalize(self):
        # Save results to final CSV
        with open('final_nail_dataset.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'liked', 'length', 'color', 'design', 'shape'])
            writer.writeheader()
            writer.writerows(self.results)
        
        messagebox.showinfo("Done", "Final dataset saved to 'final_nail_dataset.csv'!")
        self.master.quit()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1150x850")
    app = ReviewLabeler(root, 'nail_data.csv', 'raw_images')
    root.mainloop()
