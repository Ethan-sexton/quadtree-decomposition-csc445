import tkinter as tk

class GUI:
    def __init__(self):
        self.root = tk.Tk(screenName="Quad Tree Image Compression", baseName="Quad Tree Image Compression",
                          className="Quad Tree Image Compression")
        
        
        label = tk.Label(self.root, text="Test")
        scroll_bar = tk.Scrollbar(self.root)
        scroll_bar.pack()
        label.pack()

    
    def start(self):
        self.root.mainloop()