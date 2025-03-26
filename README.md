# Toxicity Detection 💬

![Gradient](https://img.shields.io/badge/Gradient-yellow%20to%20purple-blueviolet?style=flat-square)

**Short Description**: This project implements a Toxic Comment Classifier to detect toxicity in English comments using a pre-trained machine learning model.

## Live Demo
You can try the app here: [Toxicity Detection on Hugging Face Spaces](https://huggingface.co/spaces/Thilak118/Toxicity_Detection)

## Overview
This project is a web application that classifies English comments as toxic or non-toxic. It uses a pre-trained machine learning model (`toxicity.h5`) to predict the toxicity of user-provided text. The app is built with Gradio for an interactive user interface and is deployed on Hugging Face Spaces.

## Features
- Detects toxicity in English comments.
- Provides a simple Gradio-based interface for users to input text and get predictions.
- Outputs a toxicity label (e.g., "Toxic" or "Non-Toxic") with a confidence score.

## Usage Instructions
1. **Access the App**:
   - Visit the live demo at [Toxicity Detection on Hugging Face Spaces](https://huggingface.co/spaces/Thilak118/Toxicity_Detection).
2. **Enter Text**:
   - In the input textbox, enter an English comment (e.g., "You are an idiot!").
3. **Get Prediction**:
   - Click the "Predict" button (or equivalent, depending on the app’s interface).
   - The app will output:
     - The predicted label ("Toxic" or "Non-Toxic").
     - The confidence score (as a percentage).

## Examples
- **Toxic Example**:
  - Input: `You are an idiot!`
  - Prediction: `Toxic` (with confidence, e.g., 92.5%)
- **Non-Toxic Example**:
  - Input: `This is a great article, thanks for sharing!`
  - Prediction: `Non-Toxic` (with confidence, e.g., 88.7%)

## Notes
- The app runs on Hugging Face Spaces’ free tier (2 GB RAM, CPU), so it may sleep after 48 hours of inactivity. Visiting the URL will wake it up.
- The model may not always be 100% accurate. Use the predictions as a guide, not a definitive judgment.

## Model Details
- **Model**: `toxicity.h5` (a pre-trained machine learning model for toxicity detection).
- **Size**: Not specified (ensure the file size is appropriate for Hugging Face Spaces’ free tier).
- **Download Link**:
  - Viewable Link (for users): [toxicity.h5 on Google Drive](https://drive.google.com/file/d/14REfZW8UQmovqwk1DOXuPtj7bY9NuTI-/view?usp=sharing)
  - Downloadable Link (for the app): [Direct Download](https://drive.google.com/uc?id=14REfZW8UQmovqwk1DOXuPtj7bY9NuTI-)
  - Ensure the link is set to "Anyone with the link can view" to allow the app to download it during deployment.

## Installation (For Local Development)
If you want to run this app locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://huggingface.co/spaces/Thilak118/Toxicity_Detection
   cd Toxicity_Detection
