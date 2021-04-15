import cv2
import numpy as np
from PIL import Image
import os

#Path to the images stored
path = '../Images'

#Create the recognizer for the images
#Local Binary Patterns Histograms is used
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Assign the cascade classifier to be used for face detection
detector = cv2.CascadeClassifier("../Cascades/haarcascade_frontalface_default.xml")

#Get the images saved in the 'Images' folder
def getImagesAndLabels(path):
    #Get the paths for all images in the 'Images' folder
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    #For every image in 'Images' folder
    for imagePath in imagePaths:
        #Convert image to black and white????
        PIL_img = Image.open(imagePath).convert('L')
        #??????????
        img_numpy = np.array(PIL_img, 'uint8')
        #Get the id from the saved name of the image
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #Returns the faces in the 'Images' folder as a list of rectanges
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids

#Print out info to the user
print("\n [INFO] Training Faces")

#Returns two arrays, the first being the faces in the images and the ids of the images
#to be used with the trainer and later, identification
faces, ids = getImagesAndLabels(path)

#Train the recognizer based on the faces and assign the ids
recognizer.train(faces, np.array(ids))

#Write the trainer.yml file to the correct directory
recognizer.write('../Trainer/trainer.yml')

#Print exiting information to the user
print("\ [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
