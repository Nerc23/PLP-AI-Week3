# BUGGY TENSORFLOW CODE - FIND AND FIX THE ERRORS!
# This code is supposed to train a CNN on CIFAR-10 dataset but contains multiple bugs

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("Loading CIFAR-10 dataset...")

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# BUG 1: Incorrect normalization - dividing by wrong value
train_images = train_images.astype('float32') / 256  # Should be 255
test_images = test_images.astype('float32') / 256    # Should be 255

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

# BUG 2: Missing reshape for labels - CIFAR-10 labels come as (50000, 1) but should be (50000,)
# train_labels = train_labels.reshape(-1)  # This line is missing
# test_labels = test_labels.reshape(-1)    # This line is missing

# Build CNN model
model = models.Sequential([
    # BUG 3: Wrong input shape - CIFAR-10 images are 32x32x3, not 28x28x1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # BUG 4: Wrong number of output classes - CIFAR-10 has 10 classes, but using sigmoid for binary
    layers.Dense(1, activation='sigmoid')  # Should be Dense(10, activation='softmax')
])

# BUG 5: Wrong loss function - using binary crossentropy for multi-class problem
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Should be 'sparse_categorical_crossentropy'
              metrics=['accuracy'])

model.summary()

print("Training model...")

# BUG 6: Trying to fit with wrong data shapes due to previous bugs
try:
    history = model.fit(train_images, train_labels, 
                       epochs=2, 
                       batch_size=32,
                       validation_data=(test_images, test_labels),
                       verbose=1)
except Exception as e:
    print(f"Training failed with error: {e}")

# BUG 7: Evaluation will fail due to previous errors
try:
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
except Exception as e:
    print(f"Evaluation failed with error: {e}")

# BUG 8: Prediction and visualization will fail
try:
    predictions = model.predict(test_images[:5])
    
    # BUG 9: Wrong class names - using MNIST digit names for CIFAR-10
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # Should be: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(test_images[i])
        # BUG 10: Wrong prediction interpretation for binary output
        predicted_class = int(predictions[i] > 0.5)  # This assumes binary classification
        plt.title(f"True: {class_names[test_labels[i][0]]}\nPred: {class_names[predicted_class]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Visualization failed with error: {e}")

# ADDITIONAL BUG 11: Memory leak - not clearing session
# Should add: tf.keras.backend.clear_session()