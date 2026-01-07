from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

def load_data(train_path, val_path, test_path, transform_train=None, transform_val=None, transform_test=None, batch_size=32):
    """
    Fungsi untuk memuat dataset klasifikasi sayuran.
    Args:
        train_path: Path folder dataset training.
        val_path: Path folder dataset validation.
        test_path: Path folder dataset testing.
        transform_train: Transformasi untuk data training.
        transform_val: Transformasi untuk data validation.
        transform_test: Transformasi untuk data testing.
        batch_size: Ukuran batch untuk DataLoader.

    Returns:
        imdsTrain: DataLoader untuk data training.
        imdsValidation: DataLoader untuk data validation.
        imdsTest: DataLoader untuk data testing.
    """
    # Load dataset
    train_dataset = ImageFolder(train_path, transform=transform_train)
    val_dataset = ImageFolder(val_path, transform=transform_val)
    test_dataset = ImageFolder(test_path, transform=transform_test)

    # Buat DataLoader
    imdsTrain = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    imdsValidation = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    imdsTest = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Tampilkan ringkasan jumlah gambar
    print('Ringkasan Dataset:')
    print('--------------------------')
    print(f'Jumlah gambar Train      : {len(train_dataset)}')
    print(f'Jumlah gambar Validation : {len(val_dataset)}')
    print(f'Jumlah gambar Test       : {len(test_dataset)}')
    print('--------------------------')

    return imdsTrain, imdsValidation, imdsTest
