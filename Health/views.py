import logging
from django.shortcuts import render, redirect
from django.contrib.auth import logout, login
from django.contrib import messages
from Mental_Health.settings import BASE_DIR
from .models import *
from django.contrib.auth.decorators import login_required
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import librosa
from pydub import AudioSegment
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import glob
import soundfile as sf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array


IMAGE_DATASET_DIR = os.path.join(BASE_DIR, 'Dataset', 'Facial')  

# Define emotion labels based on folder names
emotion_labels = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sadness': 4,
    'surprise': 5,
    'neutral': 6
}

# Load and preprocess the image with data augmentation
def load_image_data_from_folder():
    data = []
    labels = []

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Iterate over each emotion folder
    for emotion, label in emotion_labels.items():
        emotion_folder_path = os.path.join(IMAGE_DATASET_DIR, emotion)
        
        # Iterate through each image in the folder
        for image_filename in os.listdir(emotion_folder_path):
            image_path = os.path.join(emotion_folder_path, image_filename)

            # Load and preprocess the image
            img = load_img(image_path, target_size=(224, 224))  # Increase target size for better details
            img_array = img_to_array(img)
            img_array = img_array / 255.0  # Normalize image

            data.append(img_array)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=7)  # One-hot encode labels

    return data, labels

# Modify the CNN model to use more complex layers for better feature extraction
def create_image_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions in dataset
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the image emotion model
def train_image_model():
    data, labels = load_image_data_from_folder()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    image_model = create_image_model()
    image_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
    image_model_path = os.path.join(BASE_DIR, 'models', 'image_emotion_model.h5')
    image_model.save(image_model_path)
    print("Image emotion model trained and saved successfully.")
    return image_model

# Run image model training
# train_image_model()

# Directory for audio dataset
AUDIO_DATASET_DIR = os.path.join(BASE_DIR, 'Dataset', 'Audio')

# Emotion labels (can adjust based on your dataset)
motion_labels = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sadness': 4,
    'surprise': 5,
    'neutral': 6
}

# Extract MFCC or Mel-spectrogram features from the audio files
def extract_audio_features(audio_path, use_mfcc=True):
    y, sr = librosa.load(audio_path, sr=16000)

    if use_mfcc:
        # Extract MFCC features (20 MFCCs instead of 13)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc = np.mean(mfcc, axis=1)  # Take the mean across time frames
        return mfcc
    else:
        # Extract Mel-spectrogram features
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram = np.mean(mel_spectrogram, axis=1)  # Take the mean across time frames
        return mel_spectrogram

# Load audio data and labels from folder structure
def load_audio_data_from_folder():
    features = []
    labels = []

    # Iterate through the subfolders in the AUDIO_DATASET_DIR
    for emotion_folder in os.listdir(AUDIO_DATASET_DIR):
        emotion_folder_path = os.path.join(AUDIO_DATASET_DIR, emotion_folder)
        
        # Check if it's a directory and it corresponds to a valid emotion label
        if os.path.isdir(emotion_folder_path) and emotion_folder in motion_labels:
            # Iterate through all audio files in the emotion folder
            for audio_file in os.listdir(emotion_folder_path):
                if audio_file.endswith('.wav'):  # Ensure it's a .wav file
                    audio_path = os.path.join(emotion_folder_path, audio_file)
                    mfcc = extract_audio_features(audio_path)
                    features.append(mfcc)
                    labels.append(motion_labels[emotion_folder])  # Use the folder name for label
    
    features = np.array(features)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=7)  # Update the number of classes to 7
    return features, labels

# Create LSTM model for audio emotion recognition
def create_audio_model():
    model = Sequential([
        LSTM(128, input_shape=(20, 1), return_sequences=True),  # Input shape is (20, 1) for 20 MFCCs
        LSTM(128),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions in dataset
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the audio emotion model
def train_audio_model():
    features, labels = load_audio_data_from_folder()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Reshape data to (samples, timesteps, features) for LSTM input
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
    X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension

    # Create and train the model
    audio_model = create_audio_model()
    audio_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))
    
    # Save the trained model
    audio_model_path = os.path.join(BASE_DIR, 'models', 'audio_emotion_model.h5')
    audio_model.save(audio_model_path)
    print("Audio emotion model trained and saved successfully.")
    return audio_model

# Run audio model training
# train_audio_model()

def home(request):
    return render(request, 'mentalhealth/home.html')

def mental_health(request):
    return render(request, 'mentalhealth/mental_health.html')

def analysis(request):
    return render(request, 'mentalhealth/analysis.html')

def login_view(request):
    if request.method == 'POST':
        username_or_email = request.POST.get('username')
        password = request.POST.get('password')

        # Try to get the user based on username or email
        try:
            # Check if the input is an email or username
            if '@' in username_or_email:
                user = CustomUser.objects.get(email=username_or_email)
            else:
                user = CustomUser.objects.get(username=username_or_email)

            # Authenticate the user
            if user.check_password(password):
                # Automatically log the user in
                login(request, user)
                return redirect('analysis')  # Redirect to homepage or analysis page after login
            else:
                messages.error(request, 'Invalid password, please try again.')
        except CustomUser.DoesNotExist:
            messages.error(request, 'Invalid credentials, please try again.')

    return render(request, 'mentalhealth/login.html')

def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password1 != password2:
            messages.error(request, 'Passwords do not match.')
            return redirect('signup')

        if CustomUser.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
            return redirect('signup')

        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered.')
            return redirect('signup')

        new_user = CustomUser(username=username, email=email, age=age, gender=gender)
        new_user.set_password(password1)
        new_user.save()

        # Automatically log the user in after sign up
        login(request, new_user)
        messages.success(request, 'Account created successfully! You are now logged in.')
        return redirect('analysis')  # Redirect to homepage or wherever after signup

    return render(request, 'mentalhealth/signup.html')

@login_required
def logout_view(request):
    logout(request)
    return redirect('home')
# Predefined emotion and mood labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
mood_labels = ['Neutral', 'Stress', 'Happiness']

# Logger setup
logger = logging.getLogger(__name__)

# Global model references
image_model = None
audio_model = None

# Load models lazily
def load_models():
    global image_model, audio_model
    try:
        if image_model is None:
            logger.info("Loading image model...")
            image_model_path = os.path.join(BASE_DIR, 'models', 'image_emotion_model.h5')
            if not os.path.exists(image_model_path):
                logger.error(f"Model file not found at {image_model_path}")
                raise FileNotFoundError(f"Model file not found at {image_model_path}")
            image_model = tf.keras.models.load_model(image_model_path)
            logger.info("Image model loaded successfully.")
        
        if audio_model is None:
            logger.info("Loading audio model...")
            audio_model_path = os.path.join(BASE_DIR, 'models', 'audio_emotion_model.h5')
            if not os.path.exists(audio_model_path):
                logger.error(f"Model file not found at {audio_model_path}")
                raise FileNotFoundError(f"Model file not found at {audio_model_path}")
            audio_model = tf.keras.models.load_model(audio_model_path)
            logger.info("Audio model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise ValueError("Model loading failed")

# Image preprocessing function
def process_image(image_data):
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_data)).convert("RGB").resize((224, 224))  # Ensure RGB format and proper resizing
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Audio preprocessing function
def process_audio(audio_data):
    audio_data = base64.b64decode(audio_data.split(',')[1])
    audio = AudioSegment.from_file(BytesIO(audio_data), format="webm").set_channels(1).set_frame_rate(16000)
    audio_np = np.array(audio.get_array_of_samples()) / np.max(np.abs(audio.get_array_of_samples()))

    # Extracting MFCCs (Mel Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=audio_np.astype(float), sr=16000, n_mfcc=20)  # 20 MFCCs for better feature extraction
    mfcc = np.mean(mfcc, axis=1)  # Averaging over time axis (for frame-wise MFCC features)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Ensure correct shape for model
    return mfcc

@csrf_exempt
def process_assessment(request):
    if request.method == 'POST':
        # Ensure models are loaded before using them
        try:
            load_models()
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f"Error loading models: {str(e)}"}, status=500)

        user_id = request.POST.get('user_id')
        image_data = request.POST.get('image_data')
        audio_data = request.POST.get('audio_data')

        try:
            # Process Image Data
            image_array = process_image(image_data)
            if image_model is None:
                raise ValueError("Image model is not loaded.")
            
            image_prediction = image_model.predict(image_array)
            image_emotion = np.argmax(image_prediction)
            image_emotion_label = emotion_labels[image_emotion]
            image_confidence = np.max(image_prediction)

            # Convert image_confidence to a native Python float
            image_confidence = float(image_confidence)

            # Process Audio Data
            mfcc = process_audio(audio_data)
            if audio_model is None:
                raise ValueError("Audio model is not loaded.")
            
            audio_prediction = audio_model.predict(np.expand_dims(mfcc, axis=0))
            audio_mood = np.argmax(audio_prediction)
            audio_mood_label = mood_labels[audio_mood]
            audio_confidence = np.max(audio_prediction)

            # Convert audio_confidence to a native Python float
            audio_confidence = float(audio_confidence)

            # Adjusting Confidence Thresholds for Image and Audio predictions
            image_confidence_threshold = 0.6
            audio_confidence_threshold = 0.6

            if image_confidence < image_confidence_threshold:
                image_emotion_label = "Uncertain"
            if audio_confidence < audio_confidence_threshold:
                audio_mood_label = "Uncertain"

            # Log raw predictions for debugging purposes
            logger.debug(f"Image Prediction: {image_prediction}")
            logger.debug(f"Audio Prediction: {audio_prediction}")

            # Determine Mental Health Status (simple logic based on stress)
            if image_emotion == 1 or audio_mood == 1:  # If stress detected in either modality
                mental_health_status = 'Normal'
            else:
                mental_health_status = 'Stress'

            return JsonResponse({
                'status': 'success',
                'message': 'Data received and processed successfully',
                'mental_health_status': mental_health_status,
                'image_emotion': image_emotion_label,
                'audio_mood': audio_mood_label,
                'image_confidence': image_confidence,
                'audio_confidence': audio_confidence
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f"Error: {str(e)}"}, status=400)