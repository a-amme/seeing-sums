from keras import backend as K
from keras.layers import Layer

class ToIntegerOutput(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ToIntegerOutput, self).build(input_shape)

    def call(self, x):
        return(K.cast_to_floatx((x > K.mean(x))))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)