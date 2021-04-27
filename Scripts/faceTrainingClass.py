""" 
Logan Bland
Jalen Wayt
CS 472 Project
This file trains the recognizer used to identify a face
Foundation of code from: https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348 
"""

import cv2
import numpy as np
from PIL import Image
import os


class FaceTrainer:
    def __init__(self):
        # Path to the images stored
        self.path = '../Images'
        
        # Creates a "Recognizer" that uses the Local Binary Pattern Histogram Algorithm to determine the identity 
        # of an individual. The algorithm accepts an input  image which it divides into blocks. For each block, a 
        # histogram is calculated by comparing its intensity to the intensity of the center pixel. If the intensity
        # of the pixel is greater than the intesisty of the center pixel then it is set to 1,  else it is 0. These 
        # pixels are then combined to get an 8 bit number which is known as the LBP value. After the creation of the 
        # LBP values, histograms for each region of the image is created by counting the number of similar LBP values 
        # in it. These histograms are then combined to for a single histogram which is known as the feature vector of 
        # an image.
        # The LBPH is one of three face identification techniques in OpenCV library: 
        # Fisherfaces, Eigenfaces, and Local Binary Pattern Histogram.
        # Source: https://www.ijeat.org/wp-content/uploads/papers/v8i5S/E10060585S19.pdf
        # Source: https://iq.opengenus.org/lbph-algorithm-for-face-recognition/
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # A Haar Cascade Classifier identifies an object in an image or video through machine learning. 
        # In order to create a Haar Cascade Classifier, there must be numerous positive (images that contain
        # the object to be detected) and negative (images that do not contain the object to be detected) images.
        # From these images, Haar features are extracted. These features are a single value that is formed by
        # subtracting the number of pixels under a white rectangle by the number of pixels under a black rextangle.
        # This calculation can be tedious and very time consuming. In order to decrease this time, integral images
        # are used which make it easier to calculate this difference between white and black pixels by only using 
        # four pixels. In order to ignore irrelevant features, the Adaboost algorithm is used. The Adaboost algorithm
        # combines weak classifiers into a strong classifier. Below we create three different Haar Cascade Classifiers,
        # one for recognizing a frontal face, one for recognizing a profile face, and lastly one for detecting
        # a face with a mask on
        # Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
        # Source: https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/
        # Source: https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe#:~:text=Adaboost%20helps%20you%20combine%20multiple,a%20single%20%E2%80%9Cstrong%20classifier%E2%80%9D.&text=%E2%86%92%20The%20weak%20learners%20in,on%20those%20already%20handled%20well.
        
        # Assign the cascade classifier to be used for face detection
        self.frontalFaceDetector = cv2.CascadeClassifier("../Cascades/haarcascade_frontalface_default.xml")

        # Assign the cascade classifier to be used for face detection
        self.profileFaceDetector = cv2.CascadeClassifier('../Cascades/haarcascade_profileface.xml')

    # Get the images saved in the 'Images' folder
    def getImagesAndLabels(self, path):
        # Get the paths for all images in the 'Images' folder
        imagePaths = [os.path.join(self.path,f) for f in os.listdir(self.path)]
        
        # Will hold all the faces in the images
        faceSamples = []
        
        # Will hold all the user ids of the users in the images
        ids = []
        
        # For every image in 'Images' folder
        for imagePath in imagePaths:
            # Convert image to black and white using the python imaging library
            PIL_img = Image.open(imagePath).convert('L')
            
            # Create a numpy array of black and white values in the image
            # Returns a 2D array 
            img_numpy = np.array(PIL_img, 'uint8')
            
            # Get the id from the saved name of the image
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            
            # Returns the frontal faces in the 'Images' folder as a list of rectanges
            faces = self.frontalFaceDetector.detectMultiScale(img_numpy)

            # If there is not frontal face, there must be a profile face
            if(len(faces) == 0):
                faces = self.profileFaceDetector.detectMultiScale(img_numpy)

            # Set the title of the window opened with the frame
            cv2.imshow('Training', cv2.imread(imagePath))
            
            # For each face in the faces array
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        
        return faceSamples, ids

    # Train the trainer 
    def train(self):
        # Print out info to the user
        print("\n [INFO] Training Faces")
        
        # Returns two arrays, the first being the faces in the images and the ids of the images
        # to be used with the trainer and later, identification
        faces, ids = self.getImagesAndLabels(self.path)
        
        # Train the recognizer based on the faces and assign the ids
        self.recognizer.train(faces, np.array(ids))
        
        # Write the trainer.yml file to the correct directory
        self.recognizer.write('../Trainer/trainer.yml')
        
        # Print exiting information to the user
        print("\ [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def main():
    faceTrainer = FaceTrainer()
    faceTrainer.train()

if __name__ == "__main__":
    main()