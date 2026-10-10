import json
import os

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DATA_DIR = "dataset"
MODEL_PATH = "sign_language_model.h5"


class DataAugmenter:
    @staticmethod
    def add_noise(landmarks, noise_factor=0.02):
        noise = np.random.normal(0, noise_factor, landmarks.shape)
        return landmarks + noise

    @staticmethod
    def scale(landmarks, min_scale=0.85, max_scale=1.15):
        scale = np.random.uniform(min_scale, max_scale)
        return landmarks * scale

    @staticmethod
    def translate(landmarks, max_shift=0.08):
        shift = np.random.uniform(-max_shift, max_shift, size=(3,))
        return (landmarks.reshape(-1, 3) + shift).flatten()

    @staticmethod
    def rotate(landmarks, max_angle=15):
        angle = np.radians(np.random.uniform(-max_angle, max_angle, size=3))
        cx, cy, cz = np.cos(angle)
        sx, sy, sz = np.sin(angle)

        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        rotation_matrix = rz @ ry @ rx
        landmark_matrix = landmarks.reshape(-1, 3)
        return np.dot(landmark_matrix, rotation_matrix.T).flatten()

    @staticmethod
    def augment_sample(landmarks):
        augmented = landmarks.copy()
        if np.random.rand() > 0.5:
            augmented = DataAugmenter.add_noise(augmented)
        if np.random.rand() > 0.5:
            augmented = DataAugmenter.scale(augmented)
        if np.random.rand() > 0.5:
            augmented = DataAugmenter.rotate(augmented)
        if np.random.rand() > 0.5:
            augmented = DataAugmenter.translate(augmented)
        return augmented.astype(np.float32)


class SignLanguageDataCollector:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils

        os.makedirs(data_dir, exist_ok=True)
        for letter in ALPHABET:
            os.makedirs(os.path.join(data_dir, letter), exist_ok=True)

    def collect_data(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Unable to access webcam. Please check camera permissions.")

        for letter in ALPHABET:
            print(f"\nCollecting samples for letter: {letter}")
            print("Press 'C' to capture a sample, 'N' to move to the next letter, 'Q' to quit")
            count = 0

            while count < 80:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cv2.putText(frame, f"Letter: {letter}  Sample: {count}/80", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord('n'):
                    break
                if key == ord('c') and results.multi_hand_landmarks:
                    coords = []
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        coords.extend([landmark.x, landmark.y, landmark.z])

                    file_path = os.path.join(self.data_dir, letter, f"sample_{count}.json")
                    with open(file_path, "w") as f:
                        json.dump(coords, f)
                    count += 1

        cap.release()
        cv2.destroyAllWindows()


class SignLanguageTrainer:
    def __init__(self, data_dir=DATA_DIR, model_path=MODEL_PATH):
        self.data_dir = data_dir
        self.model_path = model_path

    def load_dataset(self):
        X, y = [], []

        for letter_index, letter in enumerate(ALPHABET):
            letter_dir = os.path.join(self.data_dir, letter)
            if not os.path.exists(letter_dir):
                continue

            for file_name in os.listdir(letter_dir):
                if file_name.endswith(".json"):
                    with open(os.path.join(letter_dir, file_name), "r") as f:
                        landmarks = np.asarray(json.load(f), dtype=np.float32)
                    X.append(landmarks)
                    y.append(letter_index)

                    for _ in range(3):
                        X.append(DataAugmenter.augment_sample(landmarks))
                        y.append(letter_index)

        if not X:
            raise FileNotFoundError("No dataset found. Please collect data first.")

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)

    def create_model(self):
        model = models.Sequential(
            [
                layers.Input(shape=(63,)),
                layers.BatchNormalization(),
                layers.Dense(256, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(len(ALPHABET), activation="softmax"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self):
        X, y = self.load_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = self.create_model()
        early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

        history = model.fit(
            X_train,
            y_train,
            epochs=80,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        model.save(self.model_path)
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nModel saved to {self.model_path}")
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        return history


def main():
    print("Sign Language Recognition System")
    print("1. Collect training data")
    print("2. Train model")
    print("3. Exit")

    choice = input("Enter your choice: ")
    if choice == "1":
        collector = SignLanguageDataCollector()
        collector.collect_data()
    elif choice == "2":
        trainer = SignLanguageTrainer()
        trainer.train()
    elif choice == "3":
        return
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()


# End of file


















































































































































































































































































































































































































































































































































































n
