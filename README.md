# Sign-Language Recognizer

A Python-based sign language recognition project that detects hand gestures from a webcam or uploaded image, converts them into predicted alphabet letters, and optionally speaks the recognized text aloud.

## Features

- Real-time hand landmark tracking with MediaPipe
- Improved single-letter prediction with smoothing and confidence filtering
- Data augmentation for more robust training
- Clean Streamlit interface with camera capture and live results
- Text-to-speech output for spoken feedback
- Dataset collection and model training pipeline

## Tech stack

- Python 3.10+
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Streamlit
- pyttsx3
- scikit-learn

## Project structure

- `sign_language_recognition.py` — interactive recognition app with UI
- `sign_language_training.py` — data collection + model training
- `requirements.txt` — Python dependencies

## Setup

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the model

Collect and train the model:

```bash
python sign_language_training.py
```

This will allow you to:
- collect training samples for each alphabet letter
- train a stronger neural network
- save the model as `sign_language_model.h5`

## Running the app

```bash
streamlit run sign_language_recognition.py
```

Then:
- use the browser camera to capture a sign
- view the predicted letter and confidence
- clear the detected word or speak the text aloud

## Notes

- The model is trained on hand landmark coordinates from MediaPipe.
- A higher-quality dataset and more diverse training samples improve recognition accuracy.
- For best results, use good lighting and keep the hand centered in the frame.

## Suggested future improvements

- add full word and sentence prediction
- support numbers and common phrases
- add dataset balancing for each class
- switch to a larger sign-language vocabulary model
