# ✍️ Handwritten Digit Classification using CNN

A clean, beginner-friendly yet production-ready Convolutional Neural Network (CNN) built with **TensorFlow and Keras** to classify handwritten digits from the classic **MNIST dataset**. 

---

## 🚀 Notebook Overview
This project demonstrates the core concepts of computer vision and deep learning by implementing a multi-layer CNN pipeline. The model extracts spatial hierarchies of features (edges, shapes, loops) using convolutional and pooling layers, culminating in fully connected dense layers to accurately predict digits from `0` to `9`.

* **Dataset:** MNIST (60,000 training images, 10,000 test images of size 28x28 grayscale)
* **Framework:** TensorFlow / Keras (Keras 3 syntax compliant)
* **Final Test Accuracy:** **~97.70%** (achieved in just 5 epochs)

---

## 🛠️ Architecture & Pipeline

1. **Input Layer:** Reshaped tensors `(28, 28, 1)` with pixel normalization (`0.0` to `1.0`).
2. **Convolutional Block 1:** `Conv2D` (32 filters, $3\times3$ kernel, ReLU) + `MaxPooling2D` ($2\times2$).
3. **Convolutional Block 2:** `Conv2D` (64 filters, $3\times3$ kernel, ReLU) + `MaxPooling2D` ($2\times2$).
4. **Classification Head:** 
   - `Flatten()` layer to convert 2D maps into 1D vectors.
   - `Dense` layer (128 units, ReLU).
   - `Dropout` (0.3 rate) to prevent overfitting.
   - Output `Dense` layer (10 units, Softmax activation).

---

## 💻 Code Structure

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load and Preprocess Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 2. Build Model Architecture
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# 3. Compile & Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
