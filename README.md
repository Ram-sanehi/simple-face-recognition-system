# simple-face-recognition-system


import cv2
import numpy as np
import pyttsx3

# Load pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 1)    # Volume (0.0 to 1.0)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Dictionary to store known faces
known_faces = {}

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        roi_gray = gray[y:y+h, x:x+w]

        # Check if the face is already recognized
        recognized = False
        for name, face_encodings in known_faces.items():
            try:
                # Resize the known face image to match the size of the detected face
                resized_face = cv2.resize(face_encodings, (w, h))

                # Calculate the absolute difference between the known face and the detected face
                diff = cv2.absdiff(roi_gray, resized_face)

                # Calculate the mean squared error (MSE) as a similarity measure
                mse = np.mean(diff)

                # If the MSE is below a certain threshold, consider the face a match
                if mse < 2000:  # Adjust this threshold value as needed
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    recognized = True
                    break
            except Exception as e:
                print("Error comparing faces:", e)

        # If face is not recognized, prompt user to input their name
        if not recognized:
            cv2.putText(frame, "Please enter your name:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', frame)
            
            # Wait for user input
            name = input("Please enter your name: ")

            # Store the face along with the name
            known_faces[name] = roi_gray
            
            # Speak the name of the recognized person
            engine.say("Hello " + name)
            engine.runAndWait()

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object, close all windows, and stop the text-to-speech engine
cap.release()
cv2.destroyAllWindows()
engine.stop()
