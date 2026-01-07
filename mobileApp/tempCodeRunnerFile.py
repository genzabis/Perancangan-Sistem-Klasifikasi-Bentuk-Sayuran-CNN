import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

class VegetableClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Vegetable Classifier')
        self.root.geometry('600x400')

        self.selected_image_path = None
        self.model = None
        self.image_size = (224, 224)

        # Ganti sesuai label kelas kamu
        self.class_names = ['Bayam', 'Wortel', 'Kangkung', 'Sawi']

        self.create_widgets()
        self.load_model()

    def create_widgets(self):
        self.upload_button = tk.Button(self.root, text='Upload Gambar', command=self.upload_image)
        self.upload_button.place(x=50, y=350, width=100, height=30)

        self.predict_button = tk.Button(self.root, text='Prediksi', command=self.predict)
        self.predict_button.place(x=200, y=350, width=100, height=30)

        self.canvas = tk.Canvas(self.root, width=300, height=250, bg='white')
        self.canvas.place(x=50, y=50)

        self.result_label = tk.Label(self.root, text='Prediksi: -', font=('Arial', 14))
        self.result_label.place(x=400, y=200)

    def load_model(self):
        try:
            model_path ='models/trainedModel.h5'  # Pastikan ini sesuai dengan path model kamu
            self.model = tf.keras.models.load_model(model_path)
            print('Model berhasil dimuat.')
        except Exception as e:
            messagebox.showerror('Error', f'Gagal memuat model: {str(e)}')

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
        if file_path:
            self.selected_image_path = file_path
            img = Image.open(file_path)
            img = img.resize((300, 250))
            self.tk_image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_image)

    def predict(self):
        if not self.selected_image_path:
            messagebox.showwarning('Peringatan', 'Silakan upload gambar terlebih dahulu.')
            return

        img = Image.open(self.selected_image_path).resize(self.image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_label = self.class_names[predicted_class_index]

        self.result_label.config(text=f'Prediksi: {predicted_label}')
        print(f'Prediksi: {predicted_label}')


if __name__ == '__main__':
    root = tk.Tk()
    app = VegetableClassifierApp(root)
    root.mainloop()
