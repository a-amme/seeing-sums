import matplotlib.pyplot as plt
import numpy as np 
import tensorflow 
from tensorflow import keras
from keras import Model

def seeTest(model, dataset, filename=None, images=1):

    """
    Visualize a test of an autoencoder.
        model: the autoencoder to be tested as a keras model object
        dataset: an array of data acceptable by the model
        images: the number of inputs to be given to the model
    """

    inputs = dataset['x'][:1]
    outputs = model.predict(inputs, verbose=0)

    figure, subfigures = plt.subplots(2)
    subfigures[0].imshow(inputs[0])
    subfigures[1].imshow(outputs[0])
    plt.show()
    if filename:
        plt.savefig(filename)