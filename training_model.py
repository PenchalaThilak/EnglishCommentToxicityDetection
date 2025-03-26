import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/train.csv")

# Selecting features and labels
X = df["comment_text"]  # Text data
y = df["identity_hate"]  # Target label

# Convert text data into numerical vectors
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid overfitting
X_tfidf = vectorizer.fit_transform(X)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Convert resampled data back into a DataFrame
resampled_df = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
resampled_df["identity_hate"] = y_resampled

# Save the balanced dataset
resampled_df.to_csv("/content/drive/MyDrive/balanced_train.csv", index=False)

print("SMOTE applied successfully. Balanced dataset saved as 'balanced_train.csv'")

df = pd.read_csv("/content/drive/MyDrive/train.csv")
X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 200000  # Number of words in the vocab

vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(X.values)

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Vectorizer saved successfully as 'vectorizer.pkl'")

dataset = tf.data.Dataset.from_tensor_slices((vectorizer(X.values), y))
dataset = dataset.cache().shuffle(160000).batch(16).prefetch(8)


train = dataset.take(int(len(dataset) * .7))
val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

model = Sequential([
    Embedding(MAX_FEATURES + 1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])


model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# Train the model
history = model.fit(train, epochs=1, validation_data=val)

model.save("toxicity.h5")
print("Model saved successfully as 'toxicity.h5'")

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

model = load_model("toxicity.h5")


# Example prediction
new_text = ["You freaking suck! I am going to hit you."]
new_text_vectorized = vectorizer(new_text)
new_text_vectorized = np.array(new_text_vectorized)
new_text_vectorized = np.reshape(new_text_vectorized, (1, -1))

prediction = model.predict(new_text_vectorized)
print("Prediction:", prediction)

