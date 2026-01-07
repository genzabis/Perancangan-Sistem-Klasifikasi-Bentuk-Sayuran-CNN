from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def preprocess_data(train_dir, val_dir, test_dir, image_size=(224, 224), batch_size=32):
    """
    Fungsi untuk preprocessing dataset klasifikasi sayuran.
    Args:
        train_dir: Path ke folder data training.
        val_dir: Path ke folder data validation.
        test_dir: Path ke folder data testing.
        image_size: Ukuran gambar target (default (224, 224)).
        batch_size: Ukuran batch untuk DataLoader.

    Returns:
        augTrain: DataLoader dengan augmentasi untuk training.
        augValidation: DataLoader untuk validation (resize saja).
        augTest: DataLoader untuk testing (resize saja).
    """
    # Transformasi dan augmentasi data training
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.9, 1.1)),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Transformasi untuk validation dan testing
    test_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Load dataset
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=test_transforms)
    test_dataset = ImageFolder(test_dir, transform=test_transforms)

    # DataLoader
    augTrain = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    augValidation = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    augTest = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Preprocessing data selesai.')
    return augTrain, augValidation, augTest
