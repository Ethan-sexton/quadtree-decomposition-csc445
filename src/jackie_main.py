from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import numpy as np
import os
import time
import tkinter as tk

root = tk.Tk()
root.geometry("400x400")

file = ""
memoImages = []
memoSizes = []
threshold = 0
img_lbl = tk.Label(root, image="", bg="grey")
og_size_lbl = tk.Label(root, text="")
new_size_lbl = tk.Label(root, text="")

def uploadImage(root):
    global file
    file = filedialog.askopenfilename(title="Select a file.", filetypes=(("All images", "*.png *.jpg *.jpeg"),("PNG files", "*.png"),("JPEG files", "*.jpg *.jpeg")))
    print(f"Selected: {file}")
    tk_img = ImageTk.PhotoImage(Image.open(file))
    cv_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    if cv_img.shape[2] == 4:
        rgb_img = cv_img[:, :, :3].copy()
        alpha = cv_img[:, :, 3].copy()
    else:
        rgb_img = cv_img
        alpha = None
    global memoImages, memoSizes, img_lbl
    memoImages = []
    memoSizes = []
    size_str = f"Original size: {(os.path.getsize(file) / 1024):.2f} KB"
    og_size_lbl.config(text=size_str)
    img_lbl.config(image=tk_img)
    for i in range(255):
        img_copy = rgb_img.copy()
        quadTreeDecomposition(i, img_copy)
        if alpha is not None:
            img_copy = np.dstack((img_copy, alpha))
        memoizeImage(img_copy)
    global threshold_sldr
    threshold_sldr.config(state="normal")

    #memoizeImage here. Make an await (while awaiting make loading symbol)

def setImage(index):
    global img_lbl, new_size_lbl
    img_lbl.config(image = (memoImages[int(index)]))
    img_lbl.image = ((memoImages[int(index)]))
    new_size_lbl.config(text = memoSizes[int(index)])

def quadTreeDecomposition(variance, image):
    if variance == 0:
        return
    
    height, width = image.shape[:2]
    if height <= 2 or width <= 2: #FIXME look at more
        avg_color = np.mean(image, axis=(0, 1))
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
        quadTreeDecomposition(variance, image[0:cheight, 0:cwidth])
        quadTreeDecomposition(variance, image[0:cheight, cwidth:width])
        quadTreeDecomposition(variance, image[cheight:height, 0:cwidth])
        quadTreeDecomposition(variance, image[cheight:height, cwidth:width])

def memoizeImage(image):
    global memoImages, memoSizes
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    success, buffer = cv2.imencode('.jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    if success:
        kb_size = len(buffer) / 1024
        memoSizes.append(f"New: {kb_size:.2f} KB")
    else:
        memoSizes.append("Null")
    memoImages.append(ImageTk.PhotoImage(Image.fromarray(image)))

def saveImage():
    pil = memoImages[int(threshold)]
    file_path = filedialog.asksaveasfilename(defaultextension=".jpeg", filetypes=(("All images", "*.png *.jpg *.jpeg"), ("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")))
    if file_path:
        pil.save(file_path)

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

canvas = tk.Canvas(root)
canvas.pack(fill="both", expand=True)

scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scroll.pack(side="right", fill="y")

inner = tk.Frame(canvas)
inner.pack()

label = tk.Label(inner, text="Upload an image.")
label.pack()

btn = tk.Button(inner, text="Upload.", command=lambda: uploadImage(inner))
btn.pack()

file_lbl = tk.Label(inner, text="Supports JPEG and PNG files")
file_lbl.pack()

threshold_sldr = tk.Scale(inner, from_=0, to=254, orient="horizontal", command=setImage)
threshold_sldr.pack()
threshold_sldr.config(state="disabled")

save_btn = tk.Button(inner, text="Save", command=saveImage)
save_btn.pack()

img_lbl.pack()
og_size_lbl.pack()
new_size_lbl.pack()

root.mainloop()
#to do -
#clean up messy code
#clean up UI
#add support for other image types
#prob stop memoization or alter it. For photos like iphone pictures it takes way too much time. Or implement max file size
#implement loading spinner/progress bar
#make animation (optional)