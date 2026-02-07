import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# ===============================
# 1. Load Breast Cancer Dataset
# ===============================
dataset = sklearn.datasets.load_breast_cancer()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df["label"] = dataset.target

print("Dataset Loaded Successfully!")
print(df.head())

# ===============================
# 2. Split Features and Target
# ===============================
X = df.drop("label", axis=1)
y = df["label"]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=2
)

# ===============================
# 4. Standardize Data
# ===============================
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ===============================
# 5. Build ANN Model
# ===============================
model = keras.Sequential([
    keras.Input(shape=(30,)),
    keras.layers.Dense(20, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# ===============================
# 6. Compile Model
# ===============================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# 7. Train Model
# ===============================
print("\nTraining Started...\n")

model.fit(
    X_train_std,
    y_train,
    validation_split=0.1,
    epochs=10
)

# ===============================
# 8. Evaluate Model
# ===============================
loss, accuracy = model.evaluate(X_test_std, y_test)

print("\nModel Evaluation Complete!")
print("Test Accuracy:", accuracy)

# ===============================
# 9. Save Model
# ===============================
model.save("models/ann_model.h5")

print("\nModel Saved Successfully in models/ann_model.h5")
