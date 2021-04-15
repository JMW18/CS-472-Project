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
        # Assign the recognizer and read the trainer.yml file created by the faceTraining.py script
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('../Trainer/trainer.yml')
        
        # Get the cascade
        self.faceCascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')
        self.faceCascade1 = cv2.CascadeClassifier('../Cascades/haarcascade_profileface.xml')
        
        # Assign the font when printing the user id to the screen
        self.font = cv2.FONT_HERSHEY_PLAIN

        # Assign the name
        self.name = "Unknown"
    
    def recognize(self):
        
        # Get the Video from the Camera
        videoCapture = cv2.VideoCapture(0)
        # Set the width of the window
        videoCapture.set(3, 640)
        # Set the height of the window
        videoCapture.set(4, 480)
        minW = 0.1 * videoCapture.get(3)
        minH = 0.1 * videoCapture.get(4)

        # Show the screens until closed
        while(True):
            
            # Reads each frame
            ret, frame = videoCapture.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            # Get the front facing faces in the frame
            front_faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )

            # Get the side faces in the frame
            side_faces = self.faceCascade1.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH))
            )
            
            # Load the current users JSON file 
            if os.path.isfile("../Users/Users.json") and os.access("../Users/Users.json", os.R_OK):
                data = json.load(open("../Users/Users.json"))
            else:
                data = None
            
            # Determine the identification of the individual
            # based on the faces identified 
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
            
            #Output the results to the csv file
            self.writeResults()
                
            #Set the title of the Window opened with the frame
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
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        for (x,y,w,h) in faces:
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

    def writeResults(self):
        if not os.path.isfile("../Results/results.csv"):
            # Creates the file
            #file = open('../Results/results.csv', '+a')
            with open('../Results/results.csv', mode='w') as createFile:
                create = csv.writer(createFile, delimiter=",", quotechar='"')
                create.writerow(['Time', 'Name'])
        
        #if os.path.isfile("../Results/results.csv") and os.access("../Results/results.csv", os.R_OK):
        with open('../Results/results.csv', mode='a') as resultsFile:
            resultsWriter = csv.writer(resultsFile, delimiter=",",quotechar='"')
            resultsWriter.writerow([datetime.now().strftime("%H:%M:%S"), self.name])



def main():
    faceRecognizer = FaceRecognizer()
    faceRecognizer.recognize()

if __name__ == "__main__":
    main() 
