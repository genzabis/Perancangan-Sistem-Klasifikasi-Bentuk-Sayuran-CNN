import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
import pandas as pd

def show_dataset_info(dataset):
    """
    Menampilkan informasi jumlah gambar per kelas.

    Parameters:
    - dataset: Dataset image (ImageFolder dari torchvision atau list custom)
    """
    # Hitung jumlah gambar per kelas
    labels = [label for _, label in dataset.samples]
    label_counts = Counter(labels)
    label_names = list(label_counts.keys())
    counts = list(label_counts.values())

    # Tampilkan tabel
    print('Jumlah gambar per kelas:')
    df = pd.DataFrame({'Kelas': label_names, 'Jumlah Gambar': counts})
    print(df)

    # Plot distribusi
    plt.figure(figsize=(10, 6))
    plt.bar(label_names, counts)
    plt.title('Distribusi Data per Kelas')
    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Gambar')
    plt.xticks(rotation=45)
    plt.show()


def calculate_accuracy(predicted_labels, true_labels):
    """
    Menghitung akurasi manual (%).

    Parameters:
    - predicted_labels: List atau array label hasil prediksi
    - true_labels: List atau array label sebenarnya
    """
    correct = np.sum(np.array(predicted_labels) == np.array(true_labels))
    total = len(true_labels)
    acc = (correct / total) * 100
    print(f'Akurasi: {acc:.2f}%')
    return acc


def show_sample_images(dataset, num_images=9):
    """
    Menampilkan contoh gambar secara acak dari dataset.

    Parameters:
    - dataset: Dataset image (ImageFolder dari torchvision)
    - num_images: Jumlah gambar yang ingin ditampilkan
    """
    indices = random.sample(range(len(dataset)), num_images)
    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        plt.subplot(np.ceil(np.sqrt(num_images)), np.ceil(np.sqrt(num_images)), i + 1)
        plt.imshow(img.permute(1, 2, 0))  # Pastikan gambar sudah dalam format tensor (C, H, W)
        plt.title(f'{label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
