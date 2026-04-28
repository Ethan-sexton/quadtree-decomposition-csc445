import tkinter as tk
import tkinter.ttk as ttk
import cv2
import numpy as np
import time
import os
import PIL
from classes.QuadTree import ImageCompression
from classes.ThresholdStrategy import AverageStrategy
from tkinter import filedialog
from PIL import Image, ImageTk

class QuadTreeApp:
    def on_left_arrow(self, event):
        new_threshold = max(0, self.threshold - 1)
        self.setImage(new_threshold)
    
    def on_right_arrow(self, event):
        new_threshold = min(254, self.threshold + 1)
        self.setImage(new_threshold)
    
    def __init__(self, root):
        self.root = root
        self.root.attributes("-fullscreen", True)
        
        self.file = ""
        self.memoImages = []
        self.memoSizes = []
        self.threshold = 0
        self.page_img_labels = []
        self.page_og_size_labels = []
        self.page_new_size_labels = []
        self.page_threshold_sliders = []
        self.page_original_labels = []
        self.page_psnr_labels = []
        self.page_ratio_labels = []
        self.compressor = None
        self.page_progress_bars = []
        
        self.root.bind("<Left>", self.on_left_arrow)
        self.root.bind("<Right>", self.on_right_arrow)
        
        self.setup_ui()
    
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")
        
        page_one = self.build_page()
        page_two = self.build_page(True)
        
        notebook.add(page_one, text="Pane One")
        notebook.add(page_two, text="Pane Two")
        
        notebook.bind("<<NotebookTabChanged>>", self.reset_environment)
    
    def reset_environment(self, event=None):
        # Reset all image data and UI elements
        self.memoImages = []
        self.memoSizes = []
        self.threshold = 0
        self.compressor = None
        
        for slider in self.page_threshold_sliders:
            slider.set(0)
            slider.config(state="disabled")
        
        for img_lbl in self.page_img_labels:
            img_lbl.config(image="")
            img_lbl.image = None
        
        for orig_lbl in self.page_original_labels:
            orig_lbl.config(image="")
            orig_lbl.image = None
        
        for size_lbl in self.page_og_size_labels:
            size_lbl.config(text="")
        
        for size_lbl in self.page_new_size_labels:
            size_lbl.config(text="")
        
        for psnr_lbl in self.page_psnr_labels:
            psnr_lbl.config(text="PSNR: N/A")
        
        for ratio_lbl in self.page_ratio_labels:
            ratio_lbl.config(text="Ratio: O/C")
        
        for progress_bar in self.page_progress_bars:
            progress_bar.pack_forget()
    
    def updateWindowSize(self, image_height, image_width):
        # Can remove if fullscreen is used
        window_width = max(1200, image_width * 2 + 150)
        window_height = max(800, image_height + 300)
        self.root.geometry(f"{window_width}x{window_height}")
    
    def setImage(self, index, c=False):
        self.threshold = int(index)
        for slider in self.page_threshold_sliders:
            if int(slider.get()) != self.threshold:
                slider.set(self.threshold)
        for img_lbl in self.page_img_labels:
            if self.threshold < len(self.memoImages):
                newim = ImageTk.PhotoImage(self.memoImages[self.threshold])
                if c:
                    temp_im = Image.fromarray(self.compressor.memo[self.threshold])
                    newim = ImageTk.PhotoImage(temp_im)
                img_lbl.config(image=newim)
                img_lbl.image = newim
        for new_lbl in self.page_new_size_labels:
            if self.threshold < len(self.memoSizes):
                new_lbl.config(text=f"New: {self.memoSizes[self.threshold]:.2f} kb")
        
            self.label_psnr()
            self.label_ratio()

        if hasattr(self, 'show_tree_var') and self.show_tree_var.get() and self.compressor:
            self.update_tree_display()
    
    def toggle_tree_display(self, show_tree_var):
        if show_tree_var.get() and self.compressor and self.threshold < len(self.memoImages):
            self.update_tree_display()
        else:
            self.setImage(self.threshold, c=True)
    
    def label_psnr(self):
        for psnr_lbl in self.page_psnr_labels:
            if self.compressor and self.threshold < 255:
                psnr_val = self.compressor.psnr(self.threshold)
                psnr_lbl.config(text=f"PSNR: {psnr_val:.2f} dB")
            else:
                if self.threshold < len(self.memoImages) and len(self.memoImages) > 0:
                    original_cv = cv2.cvtColor(np.array(self.memoImages[0]), cv2.COLOR_RGB2BGR)
                    processed_cv = cv2.cvtColor(np.array(self.memoImages[self.threshold]), cv2.COLOR_RGB2BGR)
                    psnr = cv2.PSNR(original_cv, processed_cv)
                    psnr_lbl.config(text=f"PSNR: {psnr:.2f} dB")
                else:
                    psnr_lbl.config(text="PSNR: N/A")
    
    def label_ratio(self):
        for ratio_lbl in self.page_ratio_labels:
            if self.compressor and self.threshold < 255:
                ratio_val = self.compressor.get_ratio(self.threshold)
            else:
                if self.threshold < len(self.memoImages) and len(self.memoImages) > 0:
                    original_r = float(self.memoSizes[0])
                    processed_r = float(self.memoSizes[self.threshold])
                    print(original_r, processed_r)
                    ratio_val = original_r / processed_r
                else:
                    ratio_val = "O/C"
            ratio_lbl.config(text=f"Ratio: {ratio_val:.2f}")
    
    def update_tree_display(self):
        if not self.compressor:
            return
        
        # Get the image with tree overlay
        img_data = self.compressor.display(self.threshold)
        
        if img_data is None:
            return
        
        img_data = img_data.copy()
        
        # Draw rectangles on the image
        rects = self.compressor.rectangles.get(self.threshold, [])
        for xmin, xmax, ymin, ymax in rects:
            cv2.rectangle(img_data, (ymin, xmin), (ymax, xmax), (0, 255, 0), 1)
        
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        elif len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        
        # Display the image with tree overlay
        for img_lbl in self.page_img_labels:
            newim = ImageTk.PhotoImage(Image.fromarray(img_data))
            img_lbl.config(image=newim)
            img_lbl.image = newim
    
    def quadTreeDecomposition(self, variance, image):
        if variance == 0:
            return
        
        height, width = image.shape[:2]
        if height <= 2 or width <= 2:
            avg_color = np.mean(image, axis=(0, 1))
            avg_color = np.round(avg_color).astype(np.uint8)
            if np.max(np.abs(image - np.mean(image, axis=(0, 1)))) < variance:
                avg_color = np.round(avg_color).astype(np.uint8)
                image[:, :, :] = avg_color
            return
        
        cheight, cwidth = height // 2, width // 2
        
        avg_color = np.mean(image, axis=(0, 1))
        diff = np.abs(image - avg_color)

        if np.max(diff) < variance:
            avg_color = np.round(avg_color).astype(np.uint8)
            image[:, :, :] = avg_color
        else:
            self.quadTreeDecomposition(variance, image[0:cheight, 0:cwidth])
            self.quadTreeDecomposition(variance, image[0:cheight, cwidth:width])
            self.quadTreeDecomposition(variance, image[cheight:height, 0:cwidth])
            self.quadTreeDecomposition(variance, image[cheight:height, cwidth:width])
    
    def memoizeImage(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        
        success, buffer = cv2.imencode('.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if success:
            kb_size = len(buffer) / 1024
            self.memoSizes.append(kb_size)
        else:
            self.memoSizes.append(None)
        self.memoImages.append(Image.fromarray(image))
    
    def saveImage(self):
        if self.threshold < len(self.memoImages):
            pil = self.memoImages[int(self.threshold)]
            file_path = filedialog.asksaveasfilename(defaultextension=".jpeg", filetypes=(("All images", "*.png *.jpg *.jpeg"), ("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")))
            if file_path:
                pil.save(file_path, quality=50)
    
    def uploadImage(self):
        self.file = tk.filedialog.askopenfilename(title="Select a file.", filetypes=(("All images", "*.png *.jpg *.jpeg"),("PNG files", "*.png"),("JPEG files", "*.jpg *.jpeg")))
        print(f"Selected: {self.file}")
        cv_img = cv2.imread(self.file, cv2.IMREAD_UNCHANGED)
        if cv_img.shape[2] == 4:
            rgb_img = cv_img[:, :, :3].copy()
            alpha = cv_img[:, :, 3].copy()
        else:
            rgb_img = cv_img.copy()
            alpha = None
        
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        self.updateWindowSize(rgb_img.shape[0], rgb_img.shape[1])
        
        self.memoImages = []
        self.memoSizes = []
        size_str = f"Original size: {(os.path.getsize(self.file) / 1024) / 2:.2f} KB"
        
        for lbl in self.page_og_size_labels:
            lbl.config(text=size_str)
        
        original_pil = Image.fromarray(rgb_img)
        for orig_lbl in self.page_original_labels:
            orig_img = ImageTk.PhotoImage(original_pil)
            orig_lbl.config(image=orig_img)
            orig_lbl.image = orig_img
        
        # Show progress bars
        for progress_bar in self.page_progress_bars:
            progress_bar.pack(pady=5)
        
        for i in range(255):
            for progress_bar in self.page_progress_bars:
                progress_bar['value'] = i
                progress_bar.update()
            
            img_copy = rgb_img.copy()
            self.quadTreeDecomposition(i, img_copy)
            if alpha is not None:
                img_copy = np.dstack((img_copy, alpha))
            self.memoizeImage(img_copy)
        
        for progress_bar in self.page_progress_bars:
            progress_bar['value'] = 255
            progress_bar.update()
        
        # Hide progress bar after 1 second
        self.root.after(1000, self.hide_progress_bars)
        
        for slider in self.page_threshold_sliders:
            slider.config(state="normal")
            slider.set(0)
        self.setImage(0)
    
    def hide_progress_bars(self):
        for progress_bar in self.page_progress_bars:
            progress_bar.pack_forget()
    
    def uploadImageClass(self):
        self.file = tk.filedialog.askopenfilename(title="Select a file.", filetypes=(("All images", "*.png *.jpg *.jpeg"),("PNG files", "*.png"),("JPEG files", "*.jpg *.jpeg")))
        print(f"Selected: {self.file}")
        cv_img = cv2.imread(self.file, cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            return

        if len(cv_img.shape) == 3 and cv_img.shape[2] >= 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        self.compressor = ImageCompression(cv_img.copy(), AverageStrategy())

        size_str = f"Original size: {(os.path.getsize(self.file) / 1024) / 2:.2f} KB"
        for lbl in self.page_og_size_labels:
            lbl.config(text=size_str)

        original_pil = Image.fromarray(cv_img)
        for orig_lbl in self.page_original_labels:
            orig_img = ImageTk.PhotoImage(original_pil)
            orig_lbl.config(image=orig_img)
            orig_lbl.image = orig_img
        
        self.memoImages = []
        self.memoSizes = []
        
        self.compressor.compute_all_thresholds(255)
        
        for thresh in sorted(self.compressor.memo.keys()):
            img_data = self.compressor.display(thresh)
            
            if img_data is None:
                continue
            
            img_data = img_data.copy()
            
            if len(img_data.shape) == 2:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
            
            self.memoImages.append(Image.fromarray(img_data))
            
            file_size = self.compressor.get_file_size(thresh)
            print(file_size)
            self.memoSizes.append(file_size)
        
        for slider in self.page_threshold_sliders:
            slider.config(state="normal")
            slider.set(0)
        self.setImage(0, True)
    
    def build_page(self, c=False):
        frame = tk.Frame(self.root)
        
        control_frame = tk.Frame(frame)
        control_frame.pack(side="top", fill="x", padx=10, pady=10)
        
        label = tk.Label(control_frame, text="Upload an image.")
        label.pack()
        if c:
            btn = tk.Button(control_frame, text="Upload.", command=self.uploadImageClass)
        else:
            btn = tk.Button(control_frame, text="Upload.", command=self.uploadImage)
        btn.pack()
        file_lbl = tk.Label(control_frame, text="Supports JPEG and PNG files")
        file_lbl.pack()
        
        if c:
            # Show tree checkbox
            show_tree_var = tk.BooleanVar()
            show_tree_checkbox = tk.Checkbutton(control_frame, text="Show Tree Boundaries", variable=show_tree_var, command=lambda: self.toggle_tree_display(show_tree_var))
            show_tree_checkbox.pack()
            self.show_tree_var = show_tree_var
            
            # Show both buttons
            button_frame = tk.Frame(control_frame)
            button_frame.pack()
        
            save_btn = tk.Button(button_frame, text="Save", command=self.saveImage)
            save_btn.pack(side="left", padx=5)
        
            save_gif_btn = tk.Button(button_frame, text="Save as GIF", command=self.saveImageAsGif)
            save_gif_btn.pack(side="left", padx=5)
        else:
            progress_bar = ttk.Progressbar(control_frame, length=300, mode='determinate', maximum=255)
            progress_bar.pack(pady=5)
            progress_bar.pack_forget()
        
            progress_label = tk.Label(control_frame, text="Ready")
            progress_label.pack()
            progress_label.pack_forget()
            
            save_btn = tk.Button(control_frame, text="Save", command=self.saveImage)
            save_btn.pack()
        
        threshold_sldr = tk.Scale(control_frame, from_=0, to=254, orient="horizontal", command=self.setImage)
        threshold_sldr.pack()
        threshold_sldr.config(state="disabled")
                
        images_frame = tk.Frame(frame)
        images_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        
        spacer_left = tk.Frame(images_frame)
        spacer_left.pack(side="left", expand=True)
        
        orig_container = tk.Frame(images_frame)
        orig_container.pack(side="left", padx=20)
        orig_label_text = tk.Label(orig_container, text="Original", font=("Arial", 12, "bold"))
        orig_label_text.pack()
        img_lbl_orig = tk.Label(orig_container, image="", bg="grey", width=400, height=400)
        img_lbl_orig.pack()
        og_lbl = tk.Label(orig_container, text="")
        og_lbl.pack()
        
        middle_container = tk.Frame(images_frame)
        middle_container.pack(side="left", padx=20, fill="y")
        psnr_lbl = tk.Label(middle_container, text="PSNR: N/A", font=("Arial", 14, "bold"), fg="blue")
        psnr_lbl.pack(expand=True)
        ratio_lbl = tk.Label(middle_container, text="Ratio: O:P", font=("Arial", 14, "bold"), fg="blue")
        ratio_lbl.pack(expand=True)
        
        proc_container = tk.Frame(images_frame)
        proc_container.pack(side="left", padx=20)
        proc_label_text = tk.Label(proc_container, text="Compressed", font=("Arial", 12, "bold"))
        proc_label_text.pack()
        img_lbl = tk.Label(proc_container, image="", bg="grey", width=400, height=400)
        img_lbl.pack()
        
        new_lbl = tk.Label(proc_container, text="")
        new_lbl.pack()
        
        spacer_right = tk.Frame(images_frame)
        spacer_right.pack(side="left", expand=True)

        self.page_img_labels.append(img_lbl)
        self.page_og_size_labels.append(og_lbl)
        self.page_new_size_labels.append(new_lbl)
        self.page_threshold_sliders.append(threshold_sldr)
        self.page_original_labels.append(img_lbl_orig)
        self.page_psnr_labels.append(psnr_lbl)
        self.page_ratio_labels.append(ratio_lbl)
            
        if not c:
            self.page_progress_bars.append(progress_bar)

        return frame
    
    def toggle_tree_display(self, show_tree_var):
        if show_tree_var.get() and self.compressor and self.threshold < len(self.memoImages):
            self.update_tree_display()
        else:
            self.setImage(self.threshold, c=True)
    
    def update_tree_display(self):
        if not self.compressor:
            return
        
        # Get the image with tree overlay
        img_data = self.compressor.display(self.threshold)
        
        if img_data is None:
            return
        
        img_data = img_data.copy()
        
        # Draw rectangles on the image
        rects = self.compressor.rectangles.get(self.threshold, [])
        for xmin, xmax, ymin, ymax in rects:
            cv2.rectangle(img_data, (ymin, xmin), (ymax, xmax), (0, 255, 0), 2)
        if len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
        
        # Display the image with tree overlay
        for img_lbl in self.page_img_labels:
            newim = ImageTk.PhotoImage(Image.fromarray(img_data))
            img_lbl.config(image=newim)
            img_lbl.image = newim
    
    def saveImageAsGif(self):
        if not self.compressor:
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=(("GIF files", "*.gif"),))
        if file_path:
            show_tree = self.show_tree_var.get() if hasattr(self, 'show_tree_var') else False
            self.compressor.animate(path=file_path, show_tree=show_tree)
            print(f"GIF saved to {file_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = QuadTreeApp(root)
    root.mainloop()
    
    # I can also get the powerpoint sorted out tonight if we want