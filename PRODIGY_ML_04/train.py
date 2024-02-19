import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC





import random

#reopening that pickled data
pick_out = open('data.pickle','rb')
data = pickle.load(pick_out)
pick_out.close()


#random.shuffle(data)
features = []
labels = []

for feature,label in data:
    features.append(feature)
    labels.append(label)


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state = 42)


model = SVC( C = 1, kernel = 'poly', gamma = 'auto')
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}% ")
while True:
    #taking user input

    image_path = input("Enter the path of the image: ")
    gest = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    #resizing the image to match training data size

    gest = cv2.resize(gest, (50,50))

    gest = gest.flatten()

    prediction = model.predict([gest])

    #prediction = model.predict(x_test)

    

    categories = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']

    

    print("Prediction is: ",categories[prediction[0]])

    

    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.show()

    n = input("Continue again?(y/n):")
    if n in ['n','N']:
        break
    else:
        pass












