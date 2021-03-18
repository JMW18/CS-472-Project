import numpy as np
import cv2

#Get the cascade
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
#Get the Video from the Camera
videoCapture = cv2.VideoCapture(0)
#Set the width of the window
videoCapture.set(3, 640)
#Set the height of the window
videoCapture.set(4, 480)

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
cap.release()
#Destroy all the windows created
cv2.destroyAllWindows()