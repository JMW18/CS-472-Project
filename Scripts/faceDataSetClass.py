import os
import cv2
import json
import io


class FaceDataset:
    def __init__(self, username, user_id):
        # Get the cascade
        self.faceCascade = cv2.CascadeClassifier(
            '../Cascades/haarcascade_frontalface_default.xml')
       
        # Get the Video from the Camera
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
            
            # Mark the faces using a blue rectangle
            for (x, y, w, h) in faces:
                
                # Mark the identified face with a blue rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                self.count += 1
                
                # Save the image of the face to the 'Images' folder
                cv2.imwrite("../Images/User." + str(self.user_id) + '.' +
                            str(self.username) + '.' + str(self.count) + '.jpg', gray[y:y+h, x:x+w])
            
            # Set the title of the window opened with the frame
            cv2.imshow('Collecting Data', frame)

            # Gets input from keyboard and '& 0xff' is added for 64-bit machines
            k = cv2.waitKey(100) & 0xff
            # Destroys window when ESC key is hit
            if k == 27:
                break
            # Take one sample photo
            elif self.count >= 100:
                break

        # Let user know the picture(s) have been taken
        print('Pictures have been taken')

        # Save the user's ID and Name to a JSON file (e.g. '1': 'Bob')
        # If the file exists, append to it. Else, create the file
        if os.path.isfile("../Users/Users.json") and os.access("../Users/Users.json", os.R_OK):
            # Get the data from the json
            with open("../Users/Users.json") as file:
                data = json.load(file)
            # Update the data
            data.update({str(self.user_id) : str(self.username)})
            # Write the data back to the file
            with open("../Users/Users.json", "w") as file:
                json.dump(data, file)
        else:
            with io.open(os.path.join("../Users/", 'Users.json'), 'w') as db_file:
                db_file.write(json.dumps({str(self.user_id) : str(self.username)}))

        # Release the capture
        self.videoCapture.release()
        # Destroy all the windows created
        cv2.destroyAllWindows()

def main():
    faceDataset = FaceDataset("test", 0)
    faceDataset.collectData()

if __name__ == "__main__":
    main()
