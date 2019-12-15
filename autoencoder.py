import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.layers import UpSampling2D, BatchNormalization
import pickle as pkl

def makeAndTrainModel(training_epochs=1):

    """
    The name says it all
    """

    # Load data
    with open('dataset.txt', 'rb') as file:
        data = pkl.load(file, encoding='latin1')['x']

    # Parameters
    input_shape = data[0].shape

    # Make network
    # Encoder portion
    inputs = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # Decoder portion
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    net = Model(inputs, outputs)
    net.compile('adam', loss='binary_crossentropy')

    net.summary()

    # Training
    net.fit(data, data, epochs=training_epochs, verbose=1)

    return net