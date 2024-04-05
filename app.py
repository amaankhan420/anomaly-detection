import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from predictor import plot_heat_map
import numpy as np
import io


class AnomalyDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.title("Anomaly Detection")
        self.geometry("1000x600")
        self.configure(background='white')  # Background color
        self.overrideredirect(False)  # Remove window decorations

        # Left frame for project name and description
        left_frame = tk.Frame(self, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        project_name_label = tk.Label(left_frame, text="Anomaly Detection for Wood Industry", font=('Helvetica', 20, 'bold'),
                                      bg='#f0f0f0')
        project_name_label.pack(pady=(50, 20))

        description_label = tk.Label(left_frame, text="This application detects anomalies in images.",
                                     font=('Helvetica', 14), bg='#f0f0f0')
        description_label.pack()

        # Right frame for buttons and selected image
        right_frame = tk.Frame(self, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, width=300, height=300, bg='white', highlightthickness=0)
        self.canvas.pack(padx=20, pady=(50, 20))

        self.btn_select_image = tk.Button(right_frame, text="Select Image", command=self.select_image,
                                          font=('Helvetica', 14), bg='#4CAF50', fg='white', relief=tk.FLAT)
        self.btn_select_image.pack(pady=10, padx=20, ipadx=10, ipady=5, anchor=tk.CENTER)

        self.btn_process_image = tk.Button(right_frame, text="Process Image", command=self.process_image,
                                           font=('Helvetica', 14), bg='#008CBA', fg='white', relief=tk.FLAT)
        self.btn_process_image.pack(pady=10, padx=20, ipadx=10, ipady=5, anchor=tk.CENTER)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.img = Image.open(file_path)
            self.img.thumbnail((300, 300))
            self.img_tk = ImageTk.PhotoImage(self.img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def process_image(self):
        if not hasattr(self, 'img'):
            messagebox.showerror("Error", "Please select an image first.")
            return

        img_array = np.array(self.img)
        plotted_image_data = plot_heat_map(img_array)

        img = Image.open(io.BytesIO(plotted_image_data))
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)


if __name__ == "__main__":
    app = AnomalyDetectionApp()
    app.mainloop()
