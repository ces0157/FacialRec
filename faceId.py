#Kivy depednices
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#other depedncies
import cv2
import tensorflow as tf
from layers import L1Dist

import os
import numpy as np

#App layout
class CamApp(App):
    def build(self):
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint = (1, .1))
        self.verifcation_label = Label(text = "Verfication Unitiated", size_hint = (1,.1))

        #Add items to layout
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verifcation_label)

        #load's our model
        self.model = tf.keras.models.load_model('siamesemodel.h5',
                            custom_objects = {'L1Dist':L1Dist})

        #Captues video
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    #constantly updates webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        #Resize camera frame feed
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame,0).tostring()
        #formats frame so it can be used
        #by kivy
        img_texture = Texture.create(size =(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
        
        self.web_cam.texture = img_texture

    #prepoccess the images
    def preprocess(self, file_path):
        #Read the image from file path
        byte_img = tf.io.read_file(file_path)
        #loads the iamge
        img = tf.io.decode_jpeg(byte_img)
        #resizes
        img = tf.image.resize(img,(100,100))
        #re-scales
        img = img / 255.0
        return img

    #model is our siamese neural network
    #detection threshold is the threshold where our predictions is positive
    #verifcation_threshold is the threshold is the proportion of positive predictions to positive samples
    def verify(self, *args):
        detection_threshold = 0.8
        verifcation_threshold = 0.5
        
        #Write our image to file
        SAVE_PATH = os.path.join('application_data', 'input_images', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        #takes images from our folder
        for image in os.listdir(os.path.join('application_data', 'verifcation_images')):
            #grabs input image from webcam
            input_img = self.preprocess(os.path.join('application_data', 'input_images', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verifcation_images', image))
            
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis =1)))
            results.append(result)
        
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verifcation_images')))
        verified = verification > verifcation_threshold

        self.verifcation_label.text = 'Verified' if verified == True else 'Unverified'
        
        return results, verified
        
if __name__ == '__main__':
    CamApp().run()

