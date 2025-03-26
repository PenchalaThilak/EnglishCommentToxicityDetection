import numpy as np
import pickle
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

# Load the saved vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the saved model
model = load_model("toxicity.h5")

print("Model and vectorizer loaded successfully.")

# Prediction function
def predict_toxicity(text):
    text_vectorized = vectorizer([text])
    text_vectorized = np.array(text_vectorized)
    text_vectorized = np.reshape(text_vectorized, (1, -1))
    prediction = model.predict(text_vectorized)[0]
    return {"Toxic": float(prediction[0]),
            "Severe Toxic": float(prediction[1]),
            "Obscene": float(prediction[2]),
            "Threat": float(prediction[3]),
            "Insult": float(prediction[4]),
            "Identity Hate": float(prediction[5])}

# Gradio Interface
demo = gr.Interface(fn=predict_toxicity,
                    inputs=gr.Textbox(placeholder="Enter a comment..."),
                    outputs=gr.Label(),
                    title="Toxic Comment Classifier",
                    description="Enter a comment and see its toxicity levels.")

# Launch the app
demo.launch()
