import cv2
import numpy as np
from PIL import Image
import os


class FaceTrainer:
    def __init__(self):
        # Path to the images stored
        self.path = '../Images'
        
        # Create the recognizer for the images that utilizes Local Binary Patterns Histograms Algorithm.
        # See a better description in the faceRecognitionClass.py file
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Assign the cascade classifier to be used for face detection
        self.detector = cv2.CascadeClassifier("../Cascades/haarcascade_frontalface_default.xml")

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
            
            # Returns the faces in the 'Images' folder as a list of rectanges
            faces = self.detector.detectMultiScale(img_numpy)

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