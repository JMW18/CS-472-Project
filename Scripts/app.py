import tkinter
from tkinter import *
import cv2
import PIL.Image
import PIL.ImageTk
import time
import numpy as np
import os
import faceRecognitionClass
import faceDataSetClass
import faceTrainingClass

# Main file to run for GUI
class App:
    def __init__(self, window, window_title, video_source=0):
        # Creates a trainer object
        self.trainer = faceTrainingClass.FaceTrainer()
        # Creates a recognizer object
        self.recognizer = faceRecognitionClass.FaceRecognizer()

        # Get the ids used in the Images folder
        self.ids = self.trainer.getImagesAndLabels('../Images')[1]
        self.ids = np.unique(self.ids).tolist()
        
        # Sets the window to be opened of the object
        self.window = window
        # Sets the title of the window of the object
        self.window.title(window_title)
        # Sets the size of the window
        self.window.geometry("550x250")
        self.window.configure(bg = "white")
    
        # Get the logo image
        logo_img = PIL.ImageTk.PhotoImage(PIL.Image.open("../Logos/Facial_Recognition_Logo.png")) 

        self.left_frame = Frame(self.window, bg= "white")
        self.left_frame.pack(side = LEFT)
        self.right_frame =  Frame(self.window, bg= "white")
        self.right_frame.pack(side = RIGHT)

        # Creates a trainer object
        self.trainer = faceTrainingClass.FaceTrainer()
        # Creates a recognizer object
        self.recognizer = faceRecognitionClass.FaceRecognizer()

        # Add the logo
        self.image_label = tkinter.Label(self.left_frame, image = logo_img)
        self.image_label.grid(row = 0)     

        # Creates a label as a description for the entry box
        self.user_id_label = tkinter.Label(self.right_frame, text="Name of new user:")
        self.user_id_label.grid(row = 0, column = 0)

        # Label for updating the dataset
        self.update_label = tkinter.Label(self.right_frame, text="To add photos of a new user:")
        self.update_label.grid(row = 1, column = 0)

        # Label for training
        self.update_label = tkinter.Label(self.right_frame, text="To train the recognizer:")
        self.update_label.grid(row = 2, column = 0)

         # Label for recognizing
        self.update_label = tkinter.Label(self.right_frame, text="To open webcam:")
        self.update_label.grid(row = 3, column = 0)
        
        # Creates an entry box for the user to enter their ID
        self.user_id_entry = tkinter.Entry(self.right_frame)
        self.user_id_entry.grid(row = 0, column = 1)

        # Creates a dataset button that allows the user to enter a new user
        # Need to add where it gets functionalility from box
        self.dataset_button = tkinter.Button(self.right_frame, text = "Update Dataset", command = lambda: self.getUserInput())
        self.dataset_button.grid(row=1, column = 1, padx = 20, pady = 20)

        # Creates a dataset button that allows the user to enter a new user
        self.trainer_button = tkinter.Button(self.right_frame, text = "Train Data", command = lambda: self.trainer.train())
        self.trainer_button.grid(row=2, column = 1, padx = 20, pady = 20)
        
        # Creates a recognizer button to be executed
        self.recognize_button = tkinter.Button(self.right_frame, text = "Recognize", command = lambda: self.recognizer.recognize())
        self.recognize_button.grid(row=3, column = 1, padx = 20, pady = 20)
        
        # Run the GUI
        self.window.mainloop()

    def getUserInput(self):
        
        # Create a new user id frommthe newly registered user
        if (len(self.ids)-1 < 0) :
            nextID = 1
            self.ids.append([nextID])
        else :
            nextID = len(self.ids)+1
            self.ids.append([nextID])

        # Creates a data collector object and pass in the user input
        self.data_collector = faceDataSetClass.FaceDataset(self.user_id_entry.get(), nextID)

        # start collecting the data (getting the images for the new user)
        self.data_collector.collectData()
       
        # Delete the data collector object
        # del self.data_collector



App(tkinter.Tk(), "CS 472 Project")
