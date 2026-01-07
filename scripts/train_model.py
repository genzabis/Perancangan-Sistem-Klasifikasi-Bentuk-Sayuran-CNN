import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, ReLU
import matplotlib.pyplot as plt

def build_model(num_classes):
    model = Sequential([
        # Layer 1
        Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        # Layer 2
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        # Fully Connected Layer
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(train_generator, val_generator, num_classes, epochs=10):
    model = build_model(num_classes)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    
    # Plot training & validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.show()

    # Plot training & validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Progress')
    plt.legend()
    plt.show()
    
    # Simpan model
    model.save('models/trainedModel.h5')
    print('Model berhasil disimpan di models/trainedModel.h5')

    return model
