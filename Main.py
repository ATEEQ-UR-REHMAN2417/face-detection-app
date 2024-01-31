import cv2
import csv
import time
import numpy as np
from deepface import DeepFace
# from phue import Bridge  # Uncomment for LED control
import pygame  # Uncomment for audio

# # Initialize Philips Hue bridge
# hue_bridge_ip = 'your_bridge_ip'
# hue_username = 'your_hue_username'
# b = Bridge(hue_bridge_ip, username=hue_username)
# b.connect()

# Uncomment and install pygame for audio
pygame.init()
pygame.mixer.init()

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Create CSV file and write header
csv_file_path = 'emotion_data.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Person', 'Emotion'])

# List to store emotions for each person
person_emotions_list = []

start_time = time.time()
person_id_counter = 1

# Uncomment and set the path to your music file
music_file_path = 'Test.mp3'

# Uncomment and set the light ID for LED control
# light_id = 1

# Start playing music
pygame.mixer.music.load(music_file_path)
pygame.mixer.music.play()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray_frame[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the resized face image
        normalized_face = resized_face / 255.0

        # Reshape the image to match the input shape of the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # Predict emotions using the pre-trained model
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # Assign a unique identifier to each person's face
        person_key = f'person{person_id_counter}'

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'{person_key}: {emotion}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save emotion data to CSV
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.time(), person_key, emotion])

        # Append the detected emotion to the list
        person_emotions_list.append(emotion)

        # Update person_id_counter for the next person
        person_id_counter += 1

    # Display the resulting frame
    cv2.imshow('Smart Emotion Recognition System', frame)

    # Check if 10 seconds have passed
    current_time = time.time()
    if current_time - start_time >= 10:
        # Calculate the overall emotion based on all detected emotions
        if person_emotions_list:
            overall_emotion = max(set(person_emotions_list), key=person_emotions_list.count)
            person_emotions_list = []  # Reset the list for the next 10 seconds

            # Uncomment for LED control
            # color_map = {'angry': 0, 'disgust': 2000, 'fear': 4000, 'happy': 6000, 'sad': 8000, 'surprise': 10000, 'neutral': 12000}
            # hue_value = color_map.get(overall_emotion, 0)
            # b.set_light(light_id, 'hue', hue_value)

        # Reset the timer
        start_time = current_time

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if music has finished playing
    if not pygame.mixer.music.get_busy():
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
