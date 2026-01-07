import os
import shutil
import random

def split_dataset(source_folder, target_folder, train_ratio, val_ratio, test_ratio):
    """
    Fungsi untuk membagi dataset menjadi train, validation, dan test.

    Parameters:
    - source_folder: folder asal dataset (berisi subfolder per label)
    - target_folder: folder tujuan dataset terpisah
    - train_ratio: rasio data training (misal 0.7)
    - val_ratio: rasio data validation (misal 0.15)
    - test_ratio: rasio data testing (misal 0.15)
    """
    categories = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]

    for category in categories:
        source_category_path = os.path.join(source_folder, category)

        # Buat folder tujuan per kategori
        train_folder = os.path.join(target_folder, 'train', category)
        val_folder = os.path.join(target_folder, 'validation', category)
        test_folder = os.path.join(target_folder, 'test', category)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Ambil semua gambar dalam folder kategori
        image_files = [f for f in os.listdir(source_category_path) if os.path.isfile(os.path.join(source_category_path, f))]
        random.shuffle(image_files)

        # Hitung jumlah gambar per bagian
        total_images = len(image_files)
        num_train = round(train_ratio * total_images)
        num_val = round(val_ratio * total_images)

        # Bagi dataset
        train_files = image_files[:num_train]
        val_files = image_files[num_train:num_train+num_val]
        test_files = image_files[num_train+num_val:]

        # Pindahkan file
        move_files(train_files, source_category_path, train_folder)
        move_files(val_files, source_category_path, val_folder)
        move_files(test_files, source_category_path, test_folder)

        print(f'Kategori {category} berhasil dibagi.')

    print('Pembagian dataset selesai.')

def move_files(files, source_path, target_path):
    """Fungsi untuk memindahkan file."""
    for file in files:
        source_file = os.path.join(source_path, file)
        target_file = os.path.join(target_path, file)
        shutil.copy2(source_file, target_file)

# Contoh pemanggilan
# split_dataset('dataset_source', 'dataset', 0.7, 0.15, 0.15)
