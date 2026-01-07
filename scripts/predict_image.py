import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def predict_image(model_path, image_path, image_size=(224, 224), class_names=None):
    """
    Fungsi untuk memprediksi kelas dari sebuah gambar (TensorFlow version)
    Args:
        model_path: Path model CNN (.h5).
        image_path: Path gambar yang akan diprediksi.
        image_size: Ukuran input model.
        class_names: List nama-nama kelas.
    Returns:
        predicted_label: Label hasil prediksi.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f'File gambar tidak ditemukan: {image_path}')

    # Load model
    model = load_model(model_path)

    # Load dan preprocess gambar
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    img_array = img_array / 255.0  # Normalisasi

    # Visualisasi gambar
    plt.imshow(img)
    plt.title('Gambar yang Diprediksi')
    plt.axis('off')
    plt.show()

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if class_names:
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = predicted_class

    print(f'Hasil Prediksi: {predicted_label}')
    return predicted_label
