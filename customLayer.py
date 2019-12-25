from keras import backend as K
from keras.layers import Layer
import numpy as np

# This layer transforms the elements of the model's softmax output vector to 
    # zeros and ones

class ToIntegerOutput(Layer):

    def __init__(self, **kwargs):
        super(ToIntegerOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ToIntegerOutput, self).build(input_shape)

    def call(self, x):
        # Make an element of the vector one if it is greater than the mean 
            # element value
        return(K.cast_to_floatx(x > K.mean(x)))