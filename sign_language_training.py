import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import mediapipe as mp
import os
import json
from datetime import datetime
import shutil
from sklearn.model_selection import train_test_split

class SignLanguageDataCollector:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create directory structure
        os.makedirs(data_dir, exist_ok=True)
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            os.makedirs(os.path.join(data_dir, letter), exist_ok=True)

    def collect_data(self):
        """Collect training data for sign language alphabet"""
        cap = cv2.VideoCapture(0)
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            sample_count = 0
            print(f"\nCollecting data for letter {letter}")
            print("Press 'C' to capture sample, 'N' for next letter, 'Q' to quit")
            
            while sample_count < 100:  # Collect 100 samples per letter
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Display info
                cv2.putText(
                    frame,
                    f"Letter: {letter} - Samples: {sample_count}/100",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                elif key == ord('n'):
                    break
                elif key == ord('c') and results.multi_hand_landmarks:
                    # Save landmarks
                    landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Save to file
                    filename = os.path.join(
                        self.data_dir,
                        letter,
                        f"sample_{sample_count}.json"
                    )
                    with open(filename, 'w') as f:
                        json.dump(landmarks, f)
                    
                    sample_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

class SignLanguageTrainer:
    def __init__(self, data_dir="dataset", model_path="sign_language_model.h5"):
        self.data_dir = data_dir
        self.model_path = model_path

    def load_dataset(self):
        """Load and preprocess the dataset"""
        X = []
        y = []
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            letter_dir = os.path.join(self.data_dir, letter)
            if not os.path.exists(letter_dir):
                continue
                
            for sample_file in os.listdir(letter_dir):
                if sample_file.endswith('.json'):
                    with open(os.path.join(letter_dir, sample_file), 'r') as f:
                        landmarks = json.load(f)
                        X.append(landmarks)
                        y.append(ord(letter) - ord('A'))
        
        return np.array(X), np.array(y)

    def create_model(self):
        """Create the neural network model"""
        model = models.Sequential([
            layers.Input(shape=(63,)),  # 21 landmarks with x,y,z coordinates
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(26, activation='softmax')  # 26 letters
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self):
        """Train the sign language recognition model"""
        # Load dataset
        print("Loading dataset...")
        X, y = self.load_dataset()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        print("Training model...")
        model = self.create_model()
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
        
        return history

def main():
    while True:
        print("\nSign Language Recognition System")
        print("1. Collect Training Data")
        print("2. Train Model")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == '1':
            collector = SignLanguageDataCollector()
            collector.collect_data()
            
        elif choice == '2':
            trainer = SignLanguageTrainer()
            trainer.train()
            
        elif choice == '3':
            break
            
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
