import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# ===============================
# 1. Load Trained Model
# ===============================
model = keras.models.load_model("models/ann_model.h5")

# ===============================
# 2. Fit Scaler Again (No joblib)
# ===============================
dataset = sklearn.datasets.load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)

scaler = StandardScaler()
scaler.fit(X)

# ===============================
# 3. Input Data (New Patient)
# ===============================
input_data = (
    12.05,14.63,78.04,449.3,0.1031,0.09092,0.06592,0.02749,
    0.1675,0.06043,0.2636,0.7294,1.848,19.87,0.005488,
    0.01427,0.02322,0.00566,0.01428,0.002422,
    13.76,20.7,89.88,582.6,0.1494,0.2156,0.305,
    0.06548,0.2747,0.08301
)

# Convert into numpy array
input_array = np.asarray(input_data).reshape(1, -1)

# Standardize Input
input_std = scaler.transform(input_array)

# ===============================
# 4. Prediction
# ===============================
prediction = model.predict(input_std)

print("\nPrediction Value:", prediction)

if prediction[0][0] > 0.5:
    print("✅ Tumor is Benign (Safe)")
else:
    print("⚠️ Tumor is Malignant (Cancerous)")
