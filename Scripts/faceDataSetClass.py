import os
import cv2
import json
import io


class FaceDataset:
    def __init__(self, username, user_id):
        
        # A Haar Cascade Classifier identifies an object in an image or video through machine learning. 
        # In order to create a Haar Cascade Classifier, there must be numerous positive (images that contain
        # the object to be detected) and negative (images that do not contain the object to be detected) images.
        # From these images, Haar features are extracted. These features are a single value that is formed by
        # subtracting the number of pixels under a white rectangle by the number of pixels under a black rextangle.
        # This calculation can be tedious and very time consuming. In order to decrease this time, integral images
        # are used which make it easier to calculate this difference between white and black pixels by only using 
        # four pixels. In order to ignore irrelevant features, the Adaboost algorithm is used. The Adaboost algorithm
        # combines weak classifiers into a strong classifier. 
        # Source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
        # Source: https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/
        # Source: https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe#:~:text=Adaboost%20helps%20you%20combine%20multiple,a%20single%20%E2%80%9Cstrong%20classifier%E2%80%9D.&text=%E2%86%92%20The%20weak%20learners%20in,on%20those%20already%20handled%20well.
        
        # Get the frontal face cascade
        self.faceCascade = cv2.CascadeClassifier(
            '../Cascades/haarcascade_frontalface_default.xml')

        # Get the mask cascade
        self.maskCascade = cv2.CascadeClassifier(
            '../Cascades/mask_cascade.xml')
       
        # Get the video stream from the web camera
        self.videoCapture = cv2.VideoCapture(0)
        
        # Set the width of the window
        self.videoCapture.set(3, 640)
        
        # Set the height of the window
        self.videoCapture.set(4, 480)
        
        # Set the user's name and ID
        self.username = username
        self.user_id = user_id

        # Number of images saved for specific user
        self.count = 0

    def collectData(self):
        # Show the screens until closed
        while(True):
            
            # Reads each frame
            ret, frame = self.videoCapture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get the faces in the frame
            faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(20, 20)
            )

            # Get the faces with masks
            mask_faces = self.maskCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Take the pictures
            if (len(faces) > 0):
                self.getImages(faces, frame, gray)
            else:
                self.getImages(mask_faces, frame, gray)
            
            # Set the title of the window opened with the frame
            cv2.imshow('Collecting Data', frame)

            # Gets input from keyboard and '& 0xff' is added for 64-bit machines
            k = cv2.waitKey(100) & 0xff
            
            # Destroys window when ESC key is hit
            if k == 27:
                break
            # Take one-hundred photos of the individual
            elif self.count >= 100:
                break

        # Let user know the picture(s) have been taken
        print('Pictures have been taken')

        # Save the user's ID and Name to a JSON file (e.g. '1': 'Bob')
        self.saveUserID()

        # Release the capture
        self.videoCapture.release()
        # Destroy all the windows created
        cv2.destroyAllWindows()

    # Method used to save the uses's ID and name to a json file
    def saveUserID(self):
        # If the User.json is already created, append the new user to it
        if os.path.isfile("../Users/Users.json") and os.access("../Users/Users.json", os.R_OK):
            # Get the data from the User.json
            with open("../Users/Users.json") as file:
                data = json.load(file)
            # Update the data
            data.update({str(self.user_id) : str(self.username)})
            # Write the data back to the file
            with open("../Users/Users.json", "w") as file:
                json.dump(data, file)
        # Else, create the User.json file and write the first user to it
        else:
            with io.open(os.path.join("../Users/", 'Users.json'), 'w') as db_file:
                db_file.write(json.dumps({str(self.user_id) : str(self.username)}))
        
    # Used to take the images of an individual
    def getImages(self, faces, frame, gray):
        for (x, y, w, h) in faces:
            # Mark the identified face with a blue rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.count += 1
            # Save the image of the face to the 'Images' folder
            cv2.imwrite("../Images/User." + str(self.user_id) + '.' +
                str(self.username) + '.' + str(self.count) + '.jpg', gray[y:y+h, x:x+w])
            

def main():
    faceDataset = FaceDataset("test", 0)
    faceDataset.collectData()

if __name__ == "__main__":
    main()
