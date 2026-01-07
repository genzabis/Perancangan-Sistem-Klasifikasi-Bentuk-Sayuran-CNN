# Supporting Functions for Vegetable Classification App (Python Version)

import os
from PIL import Image
import tkinter as tk
from tkinter import messagebox

# Fungsi untuk validasi file gambar
def validate_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    _, ext = os.path.splitext(filename)
    return ext.lower() in valid_extensions

# Fungsi untuk membaca dan resize gambar
def load_image(filename, target_size=(224, 224)):
    img = Image.open(filename).convert('RGB')
    img = img.resize(target_size)
    return img

# Fungsi untuk menampilkan pesan alert (menggunakan tkinter)
def show_alert(message, title="Informasi"):
    root = tk.Tk()
    root.withdraw()  # Sembunyikan window utama
    messagebox.showinfo(title, message)
    root.destroy()

# Fungsi untuk memformat label prediksi agar lebih rapi
def format_label(label):
    return label.replace('_', ' ').title()
