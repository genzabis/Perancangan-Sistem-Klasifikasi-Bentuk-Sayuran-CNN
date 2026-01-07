import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model(model, test_loader, device='cuda'):
    """
    Fungsi untuk menguji akurasi model CNN.
    Args:
        model: Model CNN terlatih.
        test_loader: DataLoader untuk data testing.
        device: 'cuda' atau 'cpu'.

    Returns:
        accuracy: Akurasi model pada data testing.
    """
    print('Melakukan pengujian model...')
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f'Akurasi model pada data testing: {accuracy:.2f}%')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix (Akurasi: {accuracy:.2f}%)')
    plt.show()

    print('Pengujian model selesai.')
    return accuracy
