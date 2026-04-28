import tkinter as tk
import cv2
import numpy as np
import time
import os
import PIL
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import multiprocessing as mp

shared_rgbimg = None

#Initialize pool of mp processes
def init_pool(shared_mem, height, width, channels):
    global shared_rgbimg
    shared_rgbimg = np.frombuffer(shared_mem, dtype=np.uint8).reshape((height, width, channels))

#function for mapping to mp pool
def process_quadrant(args):
    variance, x0, x1, y0, y1 = args
    quadTreeDecomposition(variance, shared_rgbimg[x0:x1, y0:y1])

def quadTreeDecomposition(variance, image):
    if variance == 0:
        return

    height, width = image.shape[:2]
    #if too small, check against variance and return
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
        #recursively split until too small or combined
        quadTreeDecomposition(variance, image[0:cheight, 0:cwidth])
        quadTreeDecomposition(variance, image[0:cheight, cwidth:width])
        quadTreeDecomposition(variance, image[cheight:height, 0:cwidth])
        quadTreeDecomposition(variance, image[cheight:height, cwidth:width])


class recursiveQTDFrame:
    #tkinter initialization
    #frame created to pass to build function
    frame = ""
    def __init__(self, parent):
        self.file = ""
        self.memoImages = []
        self.memoSizes = []
        self.threshold = 0
        self.comp_type = 0
        self.reupload = False
        self.shared_rgbimg = None

        self.frame = tk.Frame(parent)
        self.frame.grid_columnconfigure(0, weight=1)
        self.frame.grid_columnconfigure(1, weight=1)
        self.frame.grid_rowconfigure(1, weight=1)

        #Stable preview of image
        self.placeholder_img = tk.PhotoImage(width=500, height=500)

        self.img_lbl = tk.Label(self.frame, image=self.placeholder_img, bg="grey")
        self.og_size_lbl = tk.Label(self.frame, text="")
        self.og_img_lbl = tk.Label(self.frame, text="", image=self.placeholder_img, bg="grey")
        self.psnr_lbl = tk.Label(self.frame, text="")
        self.compratio_lbl = tk.Label(self.frame, text="")
        self.new_size_lbl = tk.Label(self.frame, text="")

        self.controls = tk.Frame(self.frame)
        self.controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=8)
        self.controls.grid_columnconfigure(0, weight=1)
        self.controls.grid_columnconfigure(1, weight=1)

        label = tk.Label(self.controls, text="Upload an image.")
        label.grid(row=0, column=0, columnspan=2, pady=(0, 4))

        btn = tk.Button(self.controls, text="Upload", command=lambda: self.uploadImage(self.controls))
        btn.grid(row=1, column=0, pady=(0, 4), sticky="e", padx=(0, 6))

        reupload_btn = tk.Button(self.controls, text="Recompress", command=lambda: self.recompress(self.controls))
        reupload_btn.grid(row=1, column=1, pady=(0, 4), sticky="w", padx=(6, 0))

        file_lbl = tk.Label(self.controls, text="Supports JPEG and PNG files")
        file_lbl.grid(row=2, column=0, columnspan=2, pady=(0, 8))

        self.choice = tk.IntVar(value=0)
        tk.Radiobutton(self.controls, text="Select Variance", variable=self.choice, value=0, command=lambda: self.onSelect(0)).grid(row=3, column=0, sticky="e")
        tk.Radiobutton(self.controls, text="Variance Slider", variable=self.choice, value=1, command=lambda: self.onSelect(1)).grid(row=3, column=1, sticky="w")

        self.threshold_sldr = tk.Scale(self.controls, from_=0, to=254, orient="horizontal", command=self.setImage, length=260)
        self.threshold_sldr.grid(row=4, column=0, columnspan=2, pady=(4, 4))
        self.threshold_sldr.config(state="disabled")

        self.threshold_entry = ttk.Spinbox(self.controls, from_=0, to=254, increment=1)
        self.threshold_entry.grid(row=5, column=0, columnspan=2, pady=(4, 4))

        save_btn = tk.Button(self.controls, text="Save", command=self.saveImage)
        save_btn.grid(row=6, column=0, columnspan=2, pady=(4, 8))

        self.progress = ttk.Progressbar(self.controls, length=300, mode='determinate', maximum=255)
        self.progress.grid(row = 7, column = 0, columnspan=2, padx=2, pady=2)
        self.progress.grid_remove()
#ive noticed it doesnt work for some images well, espcially those with high contrast. let me show you one
        self.og_img_lbl.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        self.img_lbl.grid(row=1, column=1, padx=8, pady=8, sticky="nsew")
        self.og_size_lbl.grid(row=3, column=0, padx=8)
        self.new_size_lbl.grid(row=3, column=1, padx=8)
        self.psnr_lbl.grid(row=4, column=0, padx=8)
        self.compratio_lbl.grid(row=4, column=1, padx=8)


    def recompress(self, frame):
        self.reupload = True
        self.uploadImage(frame)

    def uploadImage(self, frame):
        if not self.reupload:
            self.file = tk.filedialog.askopenfilename(title="Select a file.", filetypes=(("All images", "*.png *.jpg *.jpeg"), ("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")))
            print(f"Selected: {self.file}")
        self.reupload = False
        tk_img = ImageTk.PhotoImage(Image.open(self.file).resize((500, 500), Image.Resampling.LANCZOS))
        cv_img = cv2.imread(self.file, cv2.IMREAD_UNCHANGED)
        h, w, c = cv_img.shape

        #check for png/alpha then separate. Only compute on rgb channels
        if c == 4:
            rgb_img = cv_img[:, :, :3].copy()
            alpha = cv_img[:, :, 3].copy()
        else:
            rgb_img = cv_img
            alpha = None

        self.memoImages = []
        self.memoSizes = []
        size_str = f"Original size: {(os.path.getsize(self.file) / 1024):.2f} KB"
        self.og_img_lbl.config(image=tk_img)
        self.og_img_lbl.image = tk_img
        self.og_size_lbl.config(text=size_str)

        loading_img = ImageTk.PhotoImage(Image.open("resources\\scream.jpg").resize((500, 500), Image.Resampling.LANCZOS))
        self.img_lbl.config(image=loading_img)
        self.img_lbl.image = loading_img
        frame.update_idletasks()
        shared_mem = mp.RawArray('B', h * w * 3)
        self.shared_rgbimg = np.frombuffer(shared_mem, dtype=np.uint8).reshape((h, w, 3))
        ch, cw = h // 2, w // 2
        #initialize mp pool
        with mp.Pool(processes=4, initializer=init_pool, initargs=(shared_mem, h, w, 3)) as pool:
            if self.comp_type == 1:
                self.progress.grid()
                #slider variant processes 255 different variances and stores them
                for i in range(255):
                    np.copyto(self.shared_rgbimg, rgb_img)
                    pool.map(process_quadrant, [(i, 0, ch, 0, cw), (i, 0, ch, cw, w), (i, ch, h, 0, cw), (i, ch, h, cw, w)],)
                    if alpha is not None:
                        output_image = np.dstack((self.shared_rgbimg.copy(), alpha))
                    else:
                        output_image = self.shared_rgbimg.copy()
                    self.memoizeImage(output_image)
                    print(f"Image completed at count {i}")
                    self.threshold_sldr.config(state="normal")
                    self.progress['value'] = i
                    self.progress.update()
                self.setImage(0)
                self.progress.grid_remove()
            else:
                self.memoizeImage(rgb_img.copy())
                np.copyto(self.shared_rgbimg, rgb_img)
                try:
                    i = int(self.threshold_entry.get())
                except (TypeError, ValueError):
                    i = 0
                pool.map(process_quadrant, [(i, 0, ch, 0, cw), (i, 0, ch, cw, w), (i, ch, h, 0, cw), (i, ch, h, cw, w)],)
                if alpha is not None:
                    output_image = np.dstack((self.shared_rgbimg.copy(), alpha))
                else:
                    output_image = self.shared_rgbimg.copy()
                self.memoizeImage(output_image)
                self.setImage(1)

    def setImage(self, index):
        if self.comp_type == 0:
            newim = self.memoImages[1].resize((500, 500), Image.Resampling.LANCZOS)
            newim = ImageTk.PhotoImage(newim)
            self.img_lbl.config(image=newim)
            self.img_lbl.image = newim
            self.new_size_lbl.config(text=(f"New: {self.memoSizes[1]:.2f} KB"))
        else:
            self.threshold = index
            newim = self.memoImages[int(index)].resize((500, 500), Image.Resampling.LANCZOS)
            newim = ImageTk.PhotoImage(newim)
            self.img_lbl.config(image=newim)
            self.img_lbl.image = newim
            self.new_size_lbl.config(text=(f"New: {self.memoSizes[int(index)]:.2f} KB"))
        #Peak signal to noise ratio display and compression ratio
        self.psnr()
        self.compRatio()

    #"memoize" as in to store the 255 computed images so user can move slider swiftly
    def memoizeImage(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        success, buffer = cv2.imencode('.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if success:
            kb_size = len(buffer) / 1024
            self.memoSizes.append(kb_size)
        else:
            self.memoSizes.append("Null")
        self.memoImages.append(Image.fromarray(image))

    def saveImage(self):
        if self.comp_type == 0:
            pil = self.memoImages[1]
        else:
            pil = self.memoImages[int(self.threshold)]
        file_path = filedialog.asksaveasfilename(defaultextension=".jpeg", filetypes=(("All images", "*.png *.jpg *.jpeg"), ("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")))
        if file_path:
            pil.save(file_path, quality=50)

    def onSelect(self, value):
        if value == 0:
            self.threshold_sldr.config(state='disabled')
            self.threshold_entry.config(state='active')
        else:
            self.threshold_entry.config(state='disabled')
            self.threshold_sldr.config(state='active')
            messagebox.showwarning("Long Wait Ahead!", "Depending on the size/quality of the image uploaded, this may take a prohibitively long amount of time to compute. Press \"Recompress\" to proceed, or switch to the standard single-image (Select Variance) mode.")
        self.comp_type = value

    def psnr(self):
        original = np.array(self.memoImages[0].copy())
        if self.comp_type == 0:
            processed = np.array(self.memoImages[1].copy())
        else:
            processed = np.array(self.memoImages[int(self.threshold)].copy())

        if original.ndim == 3 and original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_RGBA2RGB)
        if processed.ndim == 3 and processed.shape[2] == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)
        self.psnr_lbl.config(text=f"PSNR: {cv2.PSNR(original, processed):.4f}")

    def compRatio(self):
        original_kb = os.path.getsize(self.file) / 1024 if self.file else 1
        if self.comp_type == 0:
            processed = self.memoSizes[1]
        else:
            processed = self.memoSizes[int(self.threshold)]

        ratio = processed / original_kb
        self.compratio_lbl.config(text=f"Compression Ratio: {ratio * 100:.2f}%")

#used to construct start frame
def build(parent):
    mp.freeze_support()
    app = recursiveQTDFrame(parent)
    return app.frame