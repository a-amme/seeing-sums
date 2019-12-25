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
    key = model['key']

# Obtain and save model responses
responses = encoder.predict(testImages)

outputFile = 'test_results.txt'
headers = '{0}\t{1}\t{2}\n'.format(
          'Trial', 'Stimulus', 'ModelResponse')
with open(outputFile, 'wb') as file:
    file.write(headers)
    file.close()

for response in responses:
    for i in range(len(response)):
        if response[-1 - i] != 0:
            index = response.index(response[-1 - i])
            areaJudgment = key[index]
    trial = math.floor(responses.index(response) / 2)
    stimulus = (responses.index(response) % 2) + 1
    contents = '{0}\t{1}\t{2}\n'.format(
               trial, stimulus, areaJudgment)
    with open(outputFile, 'wb') as file:
        file.write(contents)
        file.close()