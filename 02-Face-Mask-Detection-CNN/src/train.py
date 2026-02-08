import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# -------------------------------
# Load Dataset
# -------------------------------

with_mask_dir = ("dataset\with_mask")
without_mask_dir = ( "dataset\without_mask")

with_mask_files = os.listdir(with_mask_dir)
without_mask_files = os.listdir(without_mask_dir)

data = []
labels = []

IMG_SIZE = 128

# -------------------------------
# Process With Mask Images
# -------------------------------
for img_file in with_mask_files:
    image = Image.open(os.path.join(with_mask_dir, img_file))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image.convert("RGB")
    data.append(np.array(image))
    labels.append(1)

# -------------------------------
# Process Without Mask Images
# -------------------------------
for img_file in without_mask_files:
    image = Image.open(os.path.join(without_mask_dir, img_file))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image.convert("RGB")
    data.append(np.array(image))
    labels.append(0)

# Convert to Arrays
X = np.array(data) / 255.0
Y = np.array(labels)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# -------------------------------
# CNN Model
# -------------------------------
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(),

    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(2, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    epochs=5
)

# Evaluate
loss, acc = model.evaluate(X_test, Y_test)
print("✅ Test Accuracy:", acc)

# Save Model
os.makedirs("models", exist_ok=True)
model.save("momask_detector.h5")

print("✅ Model Saved Successfully!")
