# FIXED TENSORFLOW CODE - All bugs corrected with explanations!
# This code trains a CNN on CIFAR-10 dataset with all errors fixed

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("Loading CIFAR-10 dataset...")

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# FIX 1: Correct normalization - divide by 255 to normalize pixel values to [0,1]
train_images = train_images.astype('float32') / 255.0  # Fixed: was 256, now 255
test_images = test_images.astype('float32') / 255.0    # Fixed: was 256, now 255

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")

# FIX 2: Reshape labels - CIFAR-10 labels come as (50000, 1) but need (50000,) for sparse_categorical_crossentropy
train_labels = train_labels.reshape(-1)  # Fixed: Added missing reshape
test_labels = test_labels.reshape(-1)    # Fixed: Added missing reshape

print(f"Training labels shape after reshape: {train_labels.shape}")

# Build CNN model
model = models.Sequential([
    # FIX 3: Correct input shape - CIFAR-10 images are 32x32x3 (color), not 28x28x1 (grayscale)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # Fixed: (28,28,1) -> (32,32,3)
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # FIX 4: Correct output layer - CIFAR-10 has 10 classes, need softmax activation
    layers.Dense(10, activation='softmax')  # Fixed: Dense(1, sigmoid) -> Dense(10, softmax)
])

# FIX 5: Correct loss function - use sparse_categorical_crossentropy for multi-class classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Fixed: binary_crossentropy -> sparse_categorical_crossentropy
              metrics=['accuracy'])

model.summary()

print("Training model...")

# FIX 6: Training will now work with correct data shapes and loss function
try:
    history = model.fit(train_images, train_labels, 
                       epochs=3,  # Increased epochs for better training
                       batch_size=32,
                       validation_data=(test_images, test_labels),
                       verbose=1)
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")

# FIX 7: Evaluation will now work correctly
try:
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
except Exception as e:
    print(f"Evaluation failed with error: {e}")

# FIX 8 & 9: Correct prediction and visualization
try:
    predictions = model.predict(test_images[:5])
    
    # FIX 9: Correct CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']  # Fixed: Added correct CIFAR-10 class names
    
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(test_images[i])
        # FIX 10: Correct prediction interpretation for multi-class classification
        predicted_class = np.argmax(predictions[i])  # Fixed: Use argmax for multi-class
        confidence = np.max(predictions[i])
        plt.title(f"True: {class_names[test_labels[i]]}\nPred: {class_names[predicted_class]}\nConf: {confidence:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("Visualization completed successfully!")
    
except Exception as e:
    print(f"Visualization failed with error: {e}")

# FIX 11: Clear session to prevent memory leaks
tf.keras.backend.clear_session()  # Fixed: Added to clear memory

# Additional improvements for better model performance
print("\n=== SUMMARY OF FIXES ===")
print("1. Fixed normalization: 256 -> 255")
print("2. Added label reshaping for sparse_categorical_crossentropy")
print("3. Corrected input shape: (28,28,1) -> (32,32,3)")
print("4. Fixed output layer: Dense(1, sigmoid) -> Dense(10, softmax)")
print("5. Changed loss function: binary_crossentropy -> sparse_categorical_crossentropy")
print("6. Training now works with correct shapes")
print("7. Evaluation works correctly")
print("8. Fixed class names for CIFAR-10")
print("9. Corrected prediction interpretation with argmax")
print("10. Added session clearing to prevent memory leaks")

# Bonus: Plot training history if training was successful
try:
    if 'history' in locals():
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Training history plots generated successfully!")
        
except Exception as e:
    print(f"Could not plot training history: {e}")

print("\n=== DEBUGGING TIPS FOR FUTURE ===")
print("1. Always check data shapes before training")
print("2. Ensure loss function matches your problem type")
print("3. Verify input shape matches your data")
print("4. Use appropriate activation functions")
print("5. Check that your labels are in the correct format")
print("6. Monitor for dimension mismatches in error messages")
print("7. Use meaningful variable names and class labels")
print("8. Always clear sessions in notebooks to prevent memory issues")