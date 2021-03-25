import numpy as np
import cv2
import os

#Assign the recognizer and read the trainer.yml file created by the faceTraining.py script
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('../Trainer/trainer.yml')

#Get the cascade
faceCascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')

#Assign the font when printing the user id to the screen
font = cv2.FONT_HERSHEY_PLAIN

id = 0
Logan = 'Sexy Beast'
Jalen = 'Jalen'
names = ['Unknown', Logan, Jalen]

#Get the Video from the Camera
videoCapture = cv2.VideoCapture(0)
#Set the width of the window
videoCapture.set(3, 640)
#Set the height of the window
videoCapture.set(4, 480)

minW = 0.1 * videoCapture.get(3)
minH = 0.1 * videoCapture.get(4)

#Show the screens until closed
while(True):
    #Reads each frame
    ret, frame = videoCapture.read()
    #Get the faces in the frame
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    #Mark the faces using a blue rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        id, confidence = recognizer.predict(frame[y:y+h,x:x+w])

        confidence = "   {0}%".format(round(confidence))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)
        
        roi_color = frame[y:y+h, x:x+w]
    #Set the title of the Window opened with the frame
    cv2.imshow('CS-472 Project', frame)

    #Is this even needed?
    #Gets input from keyboard? and '& 0xff' is added for 64-bit machines
    k = cv2.waitKey(30) & 0xff
    #Destroys window when ESC key is hit
    if k == 27:
        break

#Release the capture
videoCapture.release()
#Destroy all the windows created
cv2.destroyAllWindows()