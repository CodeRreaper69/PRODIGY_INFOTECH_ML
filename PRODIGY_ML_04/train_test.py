import os
import numpy as np
import cv2
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import time
import matplotlib.pyplot as plt

# Reopening the pickled data
pick_out = open('data.pickle', 'rb')
data = pickle.load(pick_out)
pick_out.close()

# Randomly shuffle the data
random.shuffle(data)

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)

model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(x_train, y_train)

# Open the video capture (0 corresponds to the default camera, you can change it to the path of your video file)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gest = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the image to match training data size
    gest = cv2.resize(gest, (50, 50))

    # Flatten the image
    gest = gest.flatten()

    # Make a prediction
    prediction = model.predict([gest])

    # Display the prediction
    categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    #time.sleep(3)
    print("Prediction is: ", categories[prediction[0]])

    # Display the frame
    cv2.imshow('Frame', frame)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the captureplt.imshow(cv2.imread(image_path), cmap='gray')
    plt.show()
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
