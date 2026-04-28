def build(parent):
    import cv2
    import numpy as np
    import os
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter import messagebox
    from PIL import Image, ImageTk

    from pages import recursiveQTDFrame
    from classes.ThresholdStrategy import AverageStrategy, RangeStrategy
    from classes.QuadTree import ImageCompression

    class QuadTreeFrame(recursiveQTDFrame):
        def __init__(self, parent):
            super().__init__(parent)

            self.qt = None
            self.max_thresh = 255
            self.cache_sizes = {}

            self.show_tree_var = tk.BooleanVar(value=False)

            tk.Checkbutton(
                self.controls,
                text="Show Tree Boundaries",
                variable=self.show_tree_var
            ).grid(row=8, column=0, columnspan=2, pady=4)

            tk.Button(
                self.controls,
                text="Animate Compression",
                command=self.animateCompression
            ).grid(row=9, column=0, columnspan=2, pady=4)

        # convert QT output → displayable image
        def render(self, thresh):
            img = self.qt.display(thresh)

            if img is None:
                raise ValueError("QuadTree returned None")

            if isinstance(img, Image.Image):
                img = np.array(img)

            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            return img

        # upload + build quadtree
        def uploadImage(self, frame):
            try:
                if not self.reupload:
                    self.file = askopenfilename(
                        title="Select image",
                        filetypes=(("Images", "*.png *.jpg *.jpeg"),)
                    )

                self.reupload = False
                if not self.file:
                    return

                cv_img = cv2.imread(self.file, cv2.IMREAD_COLOR)
                if cv_img is None:
                    messagebox.showerror("Error", "Failed to load image")
                    return

                # init quadtree (FIXED: no threshold arg)
                self.qt = ImageCompression(cv_img.copy(), RangeStrategy())
                self.cache_sizes.clear()

                # original preview
                orig_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                tk_img = ImageTk.PhotoImage(
                    Image.fromarray(orig_rgb).resize((500, 500))
                )

                self.og_img_lbl.config(image=tk_img)
                self.og_img_lbl.image = tk_img

                self.og_size_lbl.config(
                    text=f"Original: {os.path.getsize(self.file)/1024:.2f} KB"
                )

                if self.comp_type == 1:
                    self.progress.grid()
                    self.threshold_sldr.config(to=self.max_thresh)

                    for i in range(self.max_thresh + 1):
                        img = self.qt.display(i)  # builds tree incrementally

                        self.cache_sizes[i] = self.qt.get_file_size(i)

                        self.progress["value"] = i
                        self.progress.update()

                    self.progress.grid_remove()
                    self.threshold_sldr.config(state="normal")

                    self.setImage(0)

                else:
                    try:
                        thresh = int(self.threshold_entry.get() or 0)
                    except:
                        thresh = 0

                    self.cache_sizes[thresh] = self.qt.get_file_size(thresh)

                    self.setImage(thresh)

            except Exception as e:
                print("Upload error:", e)
                messagebox.showerror("Error", str(e))

        # display image + stats
        def setImage(self, index):
            if self.qt is None:
                return

            img = self.render(int(index))

            pil = Image.fromarray(img).resize((500, 500))
            tk_img = ImageTk.PhotoImage(pil)

            self.img_lbl.config(image=tk_img)
            self.img_lbl.image = tk_img

            key = int(index)

            size = self.cache_sizes.get(key, self.qt.get_file_size(key))
            self.new_size_lbl.config(text=f"New: {size:.2f} KB")

            # PSNR + ratio
            try:
                thresh = int(index) if self.comp_type == 1 else int(self.threshold_entry.get() or 0)
            except:
                thresh = 0

            self.psnr_lbl.config(text=f"PSNR: {self.qt.psnr(thresh):.4f}")

            ratio = self.qt.get_ratio(thresh)
            self.compratio_lbl.config(text=f"Compression Ratio: {ratio*100:.2f}%")

        # animation
        def animateCompression(self):
            if self.qt is None:
                messagebox.showwarning("No Image", "Upload an image first")
                return

            path = asksaveasfilename(defaultextension=".gif")
            if not path:
                return

            try:
                self.qt.compute_all_thresholds(self.max_thresh)

                self.qt.animate(
                    path=path,
                    show_tree=self.show_tree_var.get()
                )

                messagebox.showinfo("Done", f"Saved:\n{path}")

            except Exception as e:
                messagebox.showerror("Error", str(e))

    return QuadTreeFrame(parent).frame