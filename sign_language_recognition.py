import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import mediapipe as mp
import os
import json
from datetime import datetime
import pyttsx3
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class DataAugmenter:
    @staticmethod
    def add_noise(landmarks, noise_factor=0.02):
        """Add random noise to landmarks"""
        noise = np.random.normal(0, noise_factor, landmarks.shape)
        return landmarks + noise
    
    @staticmethod
    def scale(landmarks, scale_factor_range=(0.8, 1.2)):
        """Scale landmarks"""
        scale_factor = np.random.uniform(*scale_factor_range)
        return landmarks * scale_factor
    
    @staticmethod
    def rotate(landmarks, max_angle=15):
        """Rotate landmarks in 3D space"""
        angle = np.random.uniform(-max_angle, max_angle)
        theta = np.radians(angle)
        
        # Create rotation matrix
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Reshape landmarks to (21, 3) for rotation
        landmarks_3d = landmarks.reshape(-1, 3)
        rotated = np.dot(landmarks_3d, rotation_matrix)
        
        return rotated.flatten()
    
    @staticmethod
    def translate(landmarks, max_shift=0.1):
        """Translate landmarks"""
        shift = np.random.uniform(-max_shift, max_shift, 3)
        landmarks_3d = landmarks.reshape(-1, 3)
        translated = landmarks_3d + shift
        return translated.flatten()
    
    @staticmethod
    def augment_sample(landmarks):
        """Apply multiple augmentation techniques"""
        augmented = landmarks.copy()
        
        # Randomly apply augmentations
        if random.random() > 0.5:
            augmented = DataAugmenter.add_noise(augmented)
        if random.random() > 0.5:
            augmented = DataAugmenter.scale(augmented)
        if random.random() > 0.5:
            augmented = DataAugmenter.rotate(augmented)
        if random.random() > 0.5:
            augmented = DataAugmenter.translate(augmented)
            
        return augmented

class SignLanguageTrainer:
    def __init__(self, data_dir="dataset", model_path="sign_language_model.h5"):
        self.data_dir = data_dir
        self.model_path = model_path
        self.history = None
        
        # Create directories for analytics
        os.makedirs("analytics", exist_ok=True)

    def create_model(self):
        """Create enhanced neural network model"""
        model = models.Sequential([
            layers.Input(shape=(63,)),
            layers.BatchNormalization(),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(26, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_and_augment_dataset(self):
        """Load and augment the dataset"""
        X, y = [], []
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            letter_dir = os.path.join(self.data_dir, letter)
            if not os.path.exists(letter_dir):
                continue
                
            for sample_file in os.listdir(letter_dir):
                if sample_file.endswith('.json'):
                    with open(os.path.join(letter_dir, sample_file), 'r') as f:
                        landmarks = np.array(json.load(f))
                        
                        # Add original sample
                        X.append(landmarks)
                        y.append(ord(letter) - ord('A'))
                        
                        # Add augmented samples
                        for _ in range(3):  # Create 3 augmented versions
                            augmented = DataAugmenter.augment_sample(landmarks)
                            X.append(augmented)
                            y.append(ord(letter) - ord('A'))
        
        return np.array(X), np.array(y)

    def generate_analytics(self, X_train, y_train, X_test, y_test):
        """Generate training analytics"""
        # Sample distribution
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(y=y_train)
        plt.title('Training Data Distribution')
        plt.xlabel('Letter')
        plt.ylabel('Count')
        
        # Feature importance
        feature_std = np.std(X_train, axis=0)
        plt.subplot(1, 2, 2)
        plt.plot(feature_std)
        plt.title('Feature Importance (Standard Deviation)')
        plt.xlabel('Feature Index')
        plt.ylabel('Standard Deviation')
        
        plt.tight_layout()
        plt.savefig('analytics/data_analysis.png')
        plt.close()

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('analytics/training_history.png')
        plt.close()

    def train(self):
        """Train the enhanced model"""
        print("Loading and augmenting dataset...")
        X, y = self.load_and_augment_dataset()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Generate analytics
        self.generate_analytics(X_train, y_train, X_test, y_test)
        
        # Create and train model
        print("Training model...")
        model = self.create_model()
        
        # Add callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        self.history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history()
        
        # Save model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_accuracy*100:.2f}%")
        
        # Save model summary
        with open('analytics/model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

class EnhancedSignLanguageRecognizer:
    def __init__(self, model_path="sign_language_model.h5"):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load model
        self.model = models.load_model(model_path)
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        
        # Enhanced buffers
        self.prediction_buffer = []
        self.word_buffer = []
        self.sentence_buffer = []
        self.confidence_threshold = 0.8
        
        # Performance tracking
        self.fps_buffer = []
        self.last_time = datetime.now()
        
        # Analytics
        self.confusion_matrix = np.zeros((26, 26))

    def smooth_predictions(self, prediction):
        """Smooth predictions using a rolling window"""
        self.prediction_buffer.append(prediction)
        if len(self.prediction_buffer) > 5:
            self.prediction_buffer.pop(0)
        
        return np.mean(self.prediction_buffer, axis=0)

    def update_confusion_matrix(self, predicted_idx, actual_idx):
        """Update confusion matrix for analytics"""
        self.confusion_matrix[actual_idx][predicted_idx] += 1

    def save_analytics(self):
        """Save recognition analytics"""
        # Save confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            xticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            yticklabels=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        )
        plt.title('Recognition Confusion Matrix')
        plt.xlabel('Predicted Letter')
        plt.ylabel('Actual Letter')
        plt.savefig('analytics/confusion_matrix.png')
        plt.close()

    def run(self):
        """Run the enhanced recognition system"""
        cap = cv2.VideoCapture(0)
        
        print("\nEnhanced Sign Language Recognition Started")
        print("Commands:")
        print("- SPACE: Complete word")
        print("- ENTER: Complete sentence and speak")
        print("- BACKSPACE: Delete last letter")
        print("- C: Clear current word")
        print("- S: Save analytics")
        print("- Q: Quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                current_time = datetime.now()
                fps = 1.0 / (current_time - self.last_time).total_seconds()
                self.last_time = current_time
                self.fps_buffer.append(fps)
                
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks with connections
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        # Get landmarks and predict
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])
                        
                        # Make prediction with smoothing
                        raw_prediction = self.model.predict(
                            np.array(landmarks).reshape(1, -1), 
                            verbose=0
                        )[0]
                        smooth_prediction = self.smooth_predictions(raw_prediction)
                        
                        letter = chr(ord('A') + np.argmax(smooth_prediction))
                        confidence = np.max(smooth_prediction)
                        
                        if confidence > self.confidence_threshold:
                            if not self.word_buffer or self.word_buffer[-1] != letter:
                                self.word_buffer.append(letter)
                        
                        # Draw predictions
                        cv2.putText(
                            frame,
                            f"Letter: {letter} ({confidence:.2f})",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                
                # Draw current word and sentence
                cv2.putText(
                    frame,
                    f"Word: {''.join(self.word_buffer)}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )
                
                cv2.putText(
                    frame,
                    f"Sentence: {' '.join(self.sentence_buffer)}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                # Draw FPS
                avg_fps = np.mean(self.fps_buffer[-30:])
                cv2.putText(
                    frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow('Enhanced Sign Language Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Complete word
                    if self.word_buffer:
                        word = ''.join(self.word_buffer)
                        self.sentence_buffer.append(word)
                        self.word_buffer = []
                elif key == ord('\r'):  # Complete sentence and speak
                    if self.sentence_buffer:
                        sentence = ' '.join(self.sentence_buffer)
                        print(f"\nSentence: {sentence}")
                        self.engine.say(sentence)
                        self.engine.runAndWait()
                        self.sentence_buffer = []
                elif key == ord('\b'):
