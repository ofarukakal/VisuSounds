import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
import random

angryMusics = ["Metalica - Master of Puppets", "Rammstein - Du Hast", "Slipknot - Psychosocial", "System of a Down - Chop Suey", "Linkin Park - Numb", "Linkin Park - In The End", "Linkin Park - One Step Closer", "Linkin Park - Crawling", "Linkin Park - Papercut", "Linkin Park - Somewhere I Belong", "Linkin Park - What I've Done", "Linkin Park - Breaking The Habit", "Linkin Park - New Divide", "Linkin Park - Faint", "Linkin Park - Given Up", "Linkin Park - Bleed It Out", "Linkin Park - Shadow Of The Day", "Linkin Park - Leave Out All The Rest", "Linkin Park - Waiting For The End", "Linkin Park - Burn It Down", "Linkin Park - Castle Of Glass", "Linkin Park - Lost In The Echo", "Linkin Park - A Light That Never Comes", "Linkin Park - Guilty All The Same", "Linkin Park - Until It's Gone", "Linkin Park - Final Masquerade", "Linkin Park - Rebellion", "Linkin Park - Heavy", "Linkin Park - Talking To Myself", "Linkin Park - One More Light"] 
happyMusics = ["Serdar Ortac - Kara Kedi", "Serdar Ortac - Dansoz", "Tarkan - Simarik", "Mustafa Sandal - Pazara Kadar", "Tarkan - Dudu", "Gulsen - Bangir Bangir"]
sadMusics = ["Murat Gogebakan - Vurgunum", "Murat Gogebakan - Yarali", "Kivircik Ali - Gul Tukendi Ben Tukendim", "Muslum Gurses - Affet", "Muslum Gurses - Kismetim Kapanmis","Sezen Aksu - Tukenecegiz"] 

sadMusic = random.choice(sadMusics)
happyMusic = random.choice(happyMusics)
angryMusic = random.choice(angryMusics)

# load model
model = load_model("visuSound.h5")


face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'happy', 'sad')
        predicted_emotion = emotions[max_index]

        if predicted_emotion == 'angry':
            cv2.putText(test_img, 'Ofkelisin! ' + angryMusic, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif predicted_emotion == 'happy':
            cv2.putText(test_img, 'Mutlusun! ' + happyMusic, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif predicted_emotion == 'sad':
            cv2.putText(test_img, 'Uzgunsun! ' + sadMusic, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('VisuSound - Emotion Based Music Recommendation System', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows