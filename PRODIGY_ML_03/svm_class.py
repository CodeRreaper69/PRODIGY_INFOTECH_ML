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
    mypet = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    #resizing the image to match training data size

    mypet = cv2.resize(mypet, (50,50))

    mypet = mypet.flatten()

    prediction = model.predict([mypet])

    #prediction = model.predict(x_test)

    

    categories = ['cat','dog']

    

    print("Prediction is: ",categories[prediction[0]])

    #mypet = x_test[0].reshape(50,50)#reshaping the initially equalised images

    plt.imshow(cv2.imread(image_path), cmap='gray')
    plt.show()

    n = input("Continue again?(y/n):")
    if n in ['n','N']:
        break
    else:
        pass












