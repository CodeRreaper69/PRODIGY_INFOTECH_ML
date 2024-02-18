import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
#import time as t


dir = "pet"

categories = ['dog','cat']

data = []

#checking if the image is showing or not

for category in categories:
    path = os.path.join(dir, category)#for joining the directories inside the path, it will show dog and then cat folder
    label = categories.index(category)#for getting index/labels of the categories
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        pet_img = cv2.imread(imgpath,0)#reading the images
        #cv2.imshow('image',pet_img)#showing the images
        pet_img = cv2.resize(pet_img,(50,50)) #resizing all the images to 50 by 50 frame
        image = np.array(pet_img).flatten()
        #this data is the resized array containing the necessary images which we will be saving

        data.append([image, label])
        #t.sleep(10)
        #print(data)
        
        #break
    
    #break
    #t.sleep(10)
#print(path)
#print(label)
#print(len(data))   
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#converting this into data.pickle for data to get stored in a much portable way
pick_in = open('data.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
