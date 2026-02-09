import numpy as np
import tensorflow as tf
import keras
import os

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# -------------------------------
# Step 0: Create Model Folder
# -------------------------------
os.makedirs("model", exist_ok=True)

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
print("Loading IMDB Dataset...")

vocab_size = 10000  # Top 10k words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------------------
# Step 2: Padding Sequences
# -------------------------------
max_length = 200

X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)

# -------------------------------
# Step 3: Build LSTM Model
# -------------------------------
model = Sequential()

model.add(Embedding(input_dim=vocab_size,
                    output_dim=128,
                    input_length=max_length))

model.add(LSTM(128))

model.add(Dense(1, activation="sigmoid"))

# -------------------------------
# Step 4: Compile Model
# -------------------------------
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("\nModel Summary:")
print(model.summary())

# -------------------------------
# Step 5: Train Model
# -------------------------------
print("\nTraining LSTM Model...")

history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# -------------------------------
# Step 7: Save Model
# -------------------------------
model.save("model/sentiment_lstm.h5")
print("\nModel Saved Successfully!")

# -------------------------------
# Step 8: Custom Prediction Function
# -------------------------------
word_index = imdb.get_word_index()

def predict_sentiment(review):
    words = review.lower().split()

    encoded_review = []
    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            encoded_review.append(word_index[word] + 3)

    padded_review = pad_sequences([encoded_review], maxlen=max_length)

    prediction = model.predict(padded_review)[0][0]

    if prediction > 0.5:
        print("\nSentiment: POSITIVE 😊")
    else:
        print("\nSentiment: NEGATIVE 😡")

    print("Prediction Score:", prediction)


# -------------------------------
# Step 9: Test Prediction
# -------------------------------
print("\n--- Custom Review Test ---")

sample_review = "The movie was amazing and I loved it"
predict_sentiment(sample_review)
