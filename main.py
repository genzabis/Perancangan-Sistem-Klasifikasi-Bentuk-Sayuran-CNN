# Klasifikasi Sayuran Menggunakan Python
# Main Script

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print('--- Klasifikasi Sayuran Dimulai ---')

# 1. Load Dataset
print('Memuat dataset...')
train_dir = 'dataset/train'
test_dir = 'dataset/test'
val_dir = 'dataset/validation'

# Preprocessing dan augmentasi
image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)
val_generator = val_datagen.flow_from_directory(val_dir, target_size=image_size, batch_size=batch_size, class_mode='categorical')

print('Dataset berhasil dimuat.')

# 2. Buat Model CNN
print('Membuat model CNN...')
model = Sequential([
    # Layer input
    Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# 3. Kompilasi Model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Training Model
print('Training model...')
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
print('Training selesai.')

# 5. Evaluasi Model
print('Evaluasi model...')
test_loss, test_acc = model.evaluate(test_generator)
print(f'Akurasi Model: {test_acc * 100:.2f}%')

# Prediksi dan Confusion Matrix
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 6. Simpan Model
print('Menyimpan model...')
model.save('models/trainedModel.h5')
print('Model berhasil disimpan di folder models.')

# 7. Prediksi Satu Gambar (Opsional)
# from tensorflow.keras.preprocessing import image
# img_path = 'contoh_gambar.jpg'
# img = image.load_img(img_path, target_size=image_size)
# img_array = image.img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)
# prediction = model.predict(img_array)
# predicted_class = list(train_generator.class_indices.keys())[np.argmax(prediction)]
# plt.imshow(img)
# plt.title(f'Prediksi: {predicted_class}')
# plt.show()

print('--- Proses Selesai ---')
