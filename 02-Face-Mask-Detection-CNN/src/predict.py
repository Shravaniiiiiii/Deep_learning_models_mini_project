import numpy as np
import cv2
from tensorflow import keras

# Load Model
model = keras.models.load_model("momask_detector.h5")

IMG_SIZE = 128

# Input Image
image_path = input("Enter Image Path: ")

# Read Image
img = cv2.imread(image_path)

if img is None:
    print("❌ Invalid Image Path!")
    exit()

# Resize + Scale
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_scaled = img_resized / 255.0
img_reshaped = np.expand_dims(img_scaled, axis=0)

# Prediction
prediction = model.predict(img_reshaped)
pred_label = np.argmax(prediction)

# Output
if pred_label == 1:
    print("✅ Person is Wearing a Mask")
else:
    print("❌ Person is NOT Wearing a Mask")
