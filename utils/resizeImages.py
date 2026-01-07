import os
import cv2

def resize_images(source_folder, target_folder, image_size):
    """
    Fungsi untuk me-resize semua gambar dari source_folder ke target_folder.

    Parameters:
    - source_folder: folder asal dataset
    - target_folder: folder tujuan dataset hasil resize
    - image_size: ukuran target gambar (tinggi, lebar) tuple
    """
    categories = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]

    for category in categories:
        source_category_path = os.path.join(source_folder, category)
        target_category_path = os.path.join(target_folder, category)

        # Buat folder tujuan jika belum ada
        os.makedirs(target_category_path, exist_ok=True)

        image_files = [f for f in os.listdir(source_category_path) if os.path.isfile(os.path.join(source_category_path, f))]

        for image_name in image_files:
            ext = os.path.splitext(image_name)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Baca dan resize gambar
                img_path = os.path.join(source_category_path, image_name)
                img = cv2.imread(img_path)
                if img is not None:
                    resized_img = cv2.resize(img, image_size)

                    # Simpan gambar ke folder tujuan
                    target_path = os.path.join(target_category_path, image_name)
                    cv2.imwrite(target_path, resized_img)

        print(f'Folder {category} berhasil di-resize.')

    print('Proses resize gambar selesai.')

# Contoh pemanggilan
# resize_images('dataset/train', 'dataset_resized/train', (224, 224))
