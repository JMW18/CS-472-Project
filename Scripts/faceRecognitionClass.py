import numpy as np
import cv2
import os
import json
import threading
import time
import csv
from datetime import datetime

class FaceRecognizer:
    
    def __init__(self):
        
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
        
        # Loads the model and state from the trainer.yml file that was created when the user trainer the recognizer
        # based on their dataset in the local "Images" folder. To summarize, this loads in the histograms created by 
        # the user's dataset to be compared with the frame in the video source to determine the identity of an individual.
        self.recognizer.read('../Trainer/trainer.yml')
        
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
        
        # Get the frontal face cascade
        self.faceCascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')
        # Get the profile face cascade
        self.profileCascade = cv2.CascadeClassifier('../Cascades/haarcascade_profileface.xml')
        # Get the mask face cascade
        self.maskCascade = cv2.CascadeClassifier('../Cascades/mask_cascade.xml')
        
        # Assign the font when printing the user id to the screen
        self.font = cv2.FONT_HERSHEY_PLAIN

        # Assign the name
        self.name = "Unknown"
    
    # Method used to recognize an individual in the frame
    def recognize(self):
        # Get the Video from the web camera
        videoCapture = cv2.VideoCapture(0)
        # Set the width of the window
        videoCapture.set(3, 640)
        # Set the height of the window
        videoCapture.set(4, 480)
        
        # Set the minimum height and width which will be later used to determine
        # the minimum height and width of a face being detected
        minW = 0.1 * videoCapture.get(3)
        minH = 0.1 * videoCapture.get(4)

        # Show the screens until closed
        while(True):
            
            # Reads each frame
            ret, frame = videoCapture.read()
            # Converts the image in the frame to grayscale
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            # For each face cascade below here is a description of the parameters passed
            # Parameters:
            #           1. gray: The image being passed
            #           2. scaleFactor: How much the image is reduced; ours is 20%.
            #              The greater the scale the faster ther algorithm works, but
            #              has more chance of missing some faces
            #           3. minNeighboprs: How many neighbors each candidate rectangle 
            #              should have. The higher the value, the higher the quality
            #              but fewer faces can be detetcted
            #           4. minSize: Minimum possible size for a face to be detected
            # Source: https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python

            # Get the front facing faces in the frame
            front_faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )

            # Get the side faces in the frame
            side_faces = self.profileCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )

            # Get the faces wearing a mask in the frame
            mask_faces = self.maskCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )
            
            # Load the current Users.json file to get the user ID and name for each registered
            # user.
            if os.path.isfile("../Users/Users.json") and os.access("../Users/Users.json", os.R_OK):
                data = json.load(open("../Users/Users.json"))
            else:
                data = None
            
            # Determine the identification of the individual based on the faces identified by each of the
            # face cascades. 
            #if (len(mask_faces) > 0):
            #   print("Mask detected")
            #   self.determineIndividual(frame, mask_faces, data)
            if(len(front_faces) > 0 and len(side_faces) == 0):
                print("Front face recognized")
                self.determineIndividual(frame, front_faces, data)
            elif (len(front_faces) == 0 and len(side_faces) > 0):
                print("Side faces recognized")
                self.determineIndividual(frame, side_faces, data)
            elif (len(front_faces) > 0 and len(side_faces) > 0):
                print("Both front and side faces recognized")
                self.determineIndividual(frame, front_faces, data)
            else:
                print("No faces recognized")
            
            # Output the results to the csv file
            self.writeResults()
                
            # Set the title of the Window opened with the frame
            cv2.imshow('CS-472 Project', frame)

            # Gets input from keyboard? and '& 0xff' is added for 64-bit machines
            k = cv2.waitKey(30) & 0xff
            # Destroy the window when the X is clicked
            if not cv2.getWindowProperty('CS-472 Project', cv2.WND_PROP_VISIBLE):
                print("Operation Cancelled")
                break
            # Destroys window when ESC key is hit
            if k == 27:
                break

        # Release the capture when the window is closed
        videoCapture.release()
        # Destroy all the windows created when the window is closed
        cv2.destroyAllWindows()

    # Method that takes in an array of faces identified and 
    # recognizes them based on the data in the trainer
    def determineIndividual(self, frame, faces, data):
        
        # Convert the passed frame to a grayscale image
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # For all the faces in the frame
        for (x,y,w,h) in faces:
            
            # self.recognizer.predict() calculates the histogram of the faces in the frame
            # and compares it to the histogrmas in the trainer.xml. It finds the closest
            # matching one and returns the name and confidence level.
            self.name, confidence = self.recognizer.predict(gray[y:y+h,x:x+w])
            
            # The less confidence, the more likely it is that inidivdual
            if(confidence < 100):
                # Set the name if the data is not empty
                if data is not None:
                    self.name = data[str(self.name)]
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    self.name = "Unknown"
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
                confidence = " {0}%".format(round(100 - confidence))
            else:
                self.name = "Unknown"
                confidence = " {0}%".format(round(100 - confidence))
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

            cv2.putText(frame, str(self.name), (x+5, y-5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x+5, y+h-5), self.font, 1, (255, 255, 0), 1)
                    
            roi_color = frame[y:y+h, x:x+w]

    # Method used to write to a .csv file that includes the timestamps
    # the recognizer is ran and the individual's name who is recognized
    def writeResults(self):
        # If the results.csv file does not exists, create it
        if not os.path.isfile("../Results/results.csv"):
            # Creates the file
            with open('../Results/results.csv', mode='w') as createFile:
                create = csv.writer(createFile, delimiter=",", quotechar='"')
                create.writerow(['Time', 'Name'])
        
        # Appends to the .csv file with timestamps and the individual's name
        # that is currently recognized by the algorithm
        with open('../Results/results.csv', mode='a') as resultsFile:
            resultsWriter = csv.writer(resultsFile, delimiter=",",quotechar='"')
            resultsWriter.writerow([datetime.now().strftime("%H:%M:%S"), self.name])

def main():
    faceRecognizer = FaceRecognizer()
    faceRecognizer.recognize()

if __name__ == "__main__":
    main() 
