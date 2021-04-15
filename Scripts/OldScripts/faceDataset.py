import os
import cv2

#Get the cascade
faceCascade = cv2.CascadeClassifier('../Cascades/haarcascade_frontalface_default.xml')
#Get the Video from the Camera
videoCapture = cv2.VideoCapture(0)
#Set the width of the window
videoCapture.set(3, 640)
#Set the height of the window
videoCapture.set(4, 480)

#Get the id for the person's images
faceID = input('Enter the individual\'s id: ')
print('\n Now taking pictures')

#Number of images saved for specific user
count = 0
#Show the screens until closed
while(True):
    #Reads each frame
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Get the faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
    )
    #Mark the faces using a blue rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        count +=1
        #Save the image of the face to the 'Images' folder
        cv2.imwrite("../Images/User." + str(faceID) + '.' + str(count) + '.jpg', gray[y:y+h,x:x+w])
        #Set the title of the window opened with the frame
        cv2.imshow('image', frame)
    
    #Is this even needed?
    #Gets input from keyboard? and '& 0xff' is added for 64-bit machines
    k = cv2.waitKey(100) & 0xff
    #Destroys window when ESC key is hit
    if k == 27:
        break
    #Take one sample photo
    elif count >= 100:
        break 

#Let user know the picture(s) have been taken
print('Pictures have been taken')

#Release the capture
videoCapture.release()
#Destroy all the windows created
cv2.destroyAllWindows()