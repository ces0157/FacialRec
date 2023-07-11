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
        self.img1 = Image(size_hint = (1,.8))
        self.button = Button(text = "Verify", size_hint = (1, .1))
        self.verifcation = Label(text = "Verfication Unitiated", size_hint = (1,.1))


        #Add items to layout
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.button)
        layout.add_widget(self.verifcation)

        return layout

if __name__ == '__main__':
    CamApp().run()

