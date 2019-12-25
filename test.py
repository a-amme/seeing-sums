import numpy as np 
import tensorflow as tf 
import keras
import pickle as pkl
import math
from model import makeAndTrainAreaModel

existingModel = None
trainingDataset = 'training_12_23.txt'
testData = 'test_data.txt'

# Load test data
with open(testData, 'rb') as file:
    allTestData = pkl.load(file, encoding='latin1')
    testImages = allTestData['images']
    testLabels = allTestData['info']

# Load pre-trained model
if existingModel:
    pass # To be added
else:
    model = makeAndTrainAreaModel(trainingDataset)
    encoder = model['model']
    # The key is the list of reference area values. It is essential to the 
        # interpretation of the model's responses
    key = model['key']

# Obtain and save model responses
responses = encoder.predict(testImages)

# Prepare file to hold model's responses
outputFile = 'test_results.txt'
headers = '{0}\t{1}\t{2}\n'.format(
          'Trial', 'Stimulus', 'ModelResponse')
with open(outputFile, 'wb') as file:
    file.write(headers)
    file.close()

for response in responses:
    for i in range(len(response)):
        # Recall that elements of model's output are zeros and ones
        # Because the activation function of the model's second-to-last layer
            # is softmax, this layer's elements are mutually inhibitive; 
            # thus the network's output is such that the first elements are 
            # ones, and after those, all others are zero
        if response[-1 - i] != 0:
            # We use the key (reference areas) to identify the largest value 
                # that the network indicates the image's area exceeds, which 
                # is the greatest-index element with a value of one
            index = response.index(response[-1 - i])
            areaJudgment = key[index]
    # Trial number and stimulus image number
    trial = math.floor(responses.index(response) / 2)
    stimulus = (responses.index(response) % 2) + 1
    # Write model response and trial identification information to file
    contents = '{0}\t{1}\t{2}\n'.format(
               trial, stimulus, areaJudgment)
    with open(outputFile, 'wb') as file:
        file.write(contents)
        file.close()