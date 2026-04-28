from pages import build as b1
from ethan_build import build as b2
import os
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

class startFrame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.state('zoomed')
        self.bg_original = Image.open("resources\\scream.jpg")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        startFrame = tk.Frame(notebook, bg="black")
        startFrame.grid_rowconfigure(0, weight=1)
        startFrame.grid_rowconfigure(5, weight=1)
        startFrame.grid_columnconfigure(0, weight=1)
        startFrame.grid_columnconfigure(1, weight=1)
        startFrame.grid_columnconfigure(2, weight=1)

        jackieFrame = tk.Frame(notebook, bg="black")
        jackieFrame.grid_rowconfigure(0, weight=1)
        jackieFrame.grid_columnconfigure(0, weight=1)

        recursiveFrame = b1(jackieFrame)
        recursiveFrame.grid(row=0, column=0)
        
        
        ethanFrame = tk.Frame(notebook, bg="black")
        ethanFrame.grid_rowconfigure(0, weight=1)
        ethanFrame.grid_columnconfigure(0, weight=1)
        nodeFrame = b2(ethanFrame)
        nodeFrame.grid(row=0, column=0)

        notebook.add(startFrame, text="Start")
        notebook.add(jackieFrame, text="Recursive")
        notebook.add(ethanFrame, text="Nodal")

        self.bg_photo = ImageTk.PhotoImage(self.bg_original.resize((1, 1), Image.Resampling.LANCZOS))
        bg_img = tk.Label(startFrame, image=self.bg_photo)
        bg_img.image = self.bg_photo
        bg_img.place(relx=0, rely=0, relwidth=1, relheight=1)
        bg_img.lower()
        self.bg_img = bg_img
        startFrame.bind("<Configure>", self.resizeBackground)
        self.root.after(0, self.resizeBackground)

        title_img = ImageTk.PhotoImage(Image.open("resources\\scream.jpg"))
        title_lbl = tk.Label(startFrame, image = title_img, bg="black")
        title_lbl.image = title_img
        title_lbl.grid(row=0, column=0, columnspan=3, sticky="n")

        about_btn = tk.Button(startFrame, text="About", width=12, height=2, bg="black", fg="white", command=self.displayInfo)
        about_btn.grid(row=2, column=1, sticky="n")

        exit_btn = tk.Button(startFrame, text="Exit", width=12, height=2, bg="black", fg="white", command = self.root.destroy)
        exit_btn.grid(row = 0, column = 0, sticky="nw", padx=10, pady=10)

    def displayInfo(self):
        print("FIXME")

    def resizeBackground(self, event=None):
        width = max(1, event.width if event is not None else self.bg_img.winfo_width())
        height = max(1, event.height if event is not None else self.bg_img.winfo_height())
        resized = self.bg_original.resize((width, height), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(resized)
        self.bg_img.config(image=self.bg_photo)
        self.bg_img.lower()

if __name__ == "__main__":
    app = startFrame()
    app.root.mainloop()