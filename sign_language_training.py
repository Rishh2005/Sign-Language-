import os
import json
from collections import deque

import cv2
import numpy as np
import pyttsx3
import streamlit as st
import tensorflow as tf
import mediapipe as mp

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODEL_PATH = "sign_language_model.h5"
CONFIDENCE_THRESHOLD = 0.8
SMOOTHING_WINDOW = 5


@st.cache_resource
def load_model():
    """Load the trained model if it exists; otherwise create a fresh one."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(len(ALPHABET), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@st.cache_resource
def get_hands_detector():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )


def flatten_landmarks(hand_landmarks):
    """Convert 21 hand landmarks (x, y, z) into a 63-feature vector."""
    coords = []
    for landmark in hand_landmarks.landmark:
        coords.extend([landmark.x, landmark.y, landmark.z])
    return np.asarray(coords, dtype=np.float32)


def predict_letter(model, landmarks, history):
    """Predict the most likely letter with smoothing to reduce flicker."""
    prediction = model.predict(np.asarray(landmarks, dtype=np.float32).reshape(1, -1), verbose=0)[0]
    history.append(prediction)
    if len(history) > SMOOTHING_WINDOW:
        history.popleft()

    smoothed = np.mean(np.asarray(list(history), dtype=np.float32), axis=0)
    idx = int(np.argmax(smoothed))
    confidence = float(smoothed[idx])
    letter = ALPHABET[idx]
    return letter, confidence, smoothed


def render_prediction_card(letter, confidence, word, sentence):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0d1117, #111827);
            border: 1px solid #2a3a52;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        ">
            <div style="font-size: 0.9rem; color: #9ca3af;">Current prediction</div>
            <div style="font-size: 2.3rem; font-weight: 700; color: #60a5fa; margin-top: 0.25rem;">{letter}</div>
            <div style="margin-top: 0.5rem; font-size: 0.95rem; color: #cbd5e1;">Confidence: {confidence:.2%}</div>
            <div style="margin-top: 0.35rem; font-size: 0.9rem; color: #a7f3d0;">Word: {word}</div>
            <div style="margin-top: 0.2rem; font-size: 0.9rem; color: #f9a8d4;">Sentence: {sentence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Sign Language Recognizer", page_icon="🤟", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        div[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.9);
        }
        .block-container {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤟 Sign Language Recognizer")
    st.caption("Better detection, smoother recognition, and a cleaner interface for sign language interpretation.")

    if "word_buffer" not in st.session_state:
        st.session_state.word_buffer = []
    if "sentence_buffer" not in st.session_state:
        st.session_state.sentence_buffer = []
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=SMOOTHING_WINDOW)
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = "-"
    if "last_confidence" not in st.session_state:
        st.session_state.last_confidence = 0.0

    model = load_model()
    hands = get_hands_detector()
    engine = pyttsx3.init()

    with st.sidebar:
        st.header("Controls")
        confidence_threshold = st.slider("Confidence threshold", 0.50, 0.99, value=CONFIDENCE_THRESHOLD, step=0.01)
        st.write("Recommended value: 0.75-0.85")

        if st.button("Clear current word"):
            st.session_state.word_buffer = []

        if st.button("Add word to sentence"):
            if st.session_state.word_buffer:
                word = "".join(st.session_state.word_buffer)
                st.session_state.sentence_buffer.append(word)
                st.session_state.word_buffer = []

        if st.button("Speak sentence"):
            if st.session_state.sentence_buffer:
                sentence = " ".join(st.session_state.sentence_buffer)
                engine.say(sentence)
                engine.runAndWait()

        if st.button("Reset sentence"):
            st.session_state.sentence_buffer = []
            st.session_state.word_buffer = []

    left_col, right_col = st.columns([1.6, 1])

    with left_col:
        uploaded_image = st.camera_input("Capture a sign")

        if uploaded_image is not None:
            image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            annotated = image.copy()
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

                    landmarks = flatten_landmarks(hand)
                    letter, confidence, _ = predict_letter(model, landmarks, st.session_state.history)
                    st.session_state.last_prediction = letter
                    st.session_state.last_confidence = confidence

                    if confidence >= confidence_threshold:
                        if not st.session_state.word_buffer or st.session_state.word_buffer[-1] != letter:
                            st.session_state.word_buffer.append(letter)

                    cv2.putText(
                        annotated,
                        f"{letter} ({confidence:.2f})",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

            st.image(annotated, channels="BGR", caption="Detected hand landmarks")
        else:
            st.info("Use the camera to capture a sign. For best results, keep the hand centered and well lit.")

    with right_col:
        word_text = "".join(st.session_state.word_buffer)
        sentence_text = " ".join(st.session_state.sentence_buffer)
        render_prediction_card(st.session_state.last_prediction, st.session_state.last_confidence, word_text, sentence_text)

        st.markdown("### Current word")
        st.code(word_text if word_text else "No current word", language="text")

        st.markdown("### Full sentence")
        st.code(sentence_text if sentence_text else "No sentence yet", language="text")

        if st.button("Speak detected word"):
            if word_text:
                engine.say(word_text)
                engine.runAndWait()

    st.markdown("---")
    st.markdown(
        """
        ### Tips for better accuracy
        - Keep your hand in the frame and avoid extreme angles.
        - Use good lighting.
        - Only one hand should be visible.
        - Try to keep the camera steady.
        """
    )


if __name__ == "__main__":
    main()
