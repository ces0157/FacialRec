#Holds the Custom L1 Distance layer module
#Needed to load the custom model

#Import depedncies

import tensorflow as tf
from keras.layers import Layer

#Custom L1 distance layer
#uses the two neural network streams
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    #Uses anchor image and positive/negative
    #to compare their similarity
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)