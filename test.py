import numpy as np 
import tensorflow as tf 
import keras
import pickle as pkl
import math
from model import makeAndTrainAreaModel
from keras.models import load_model

# test('consoliData.txt', 'test_data.txt', 1, 1)

def test(fileNameTag, testData, trainData=None, aeEpochs=None, 
         classifierEpochs=None, modelFile=None, keyFile=None):

    """
    Present a set of images to the model and record its area judgments.
    PARAMETERS:
    Note that the function will throw an error unless all of trainData, 
    aeEpochs, and classifierEpochs or all of modelFile and keyFile are provided.
    If modelFile and keyFile are given, the test will proceed using a saved model.
    Otherwise, if trainData, aeEpochs, and classifierEpochs are given, 
    the test function will create a new model.
        fileNameTag: str from which filenames for model and key (if applicable), 
            and test results will be generated
        trainData: str, the name of the file containing the training datset
        testData: str, the name of the file containing the test datset
            (only necessary if modelFile and keyFile are none)
        aeEpochs: int, the number of epochs for which to train a new model
            (only necessary if modelFile and keyFile are none)
        classifierEpochs: int, the number of epochs for which to train the 
            classifier portion of a new model (only necessary if modelFile and 
            keyFile are none)
        modelFile: str, the name of a file containing a saved model
        keyFile: str, the name of a file containing the key corresponding to the 
            saved model at modelFile
    
    """

    # Ensure arguments passed to function allow either for the use of a pre-trained 
        # model or the generation of a new one
    mode = None
    if None not in [modelFile, keyFile]:
        mode = 'pre-trained model'
    elif None not in [trainData, aeEpochs, classifierEpochs]:
        mode = 'fresh model'
    assert mode != None, 'Insufficient information provided. Check arguments to function call.'
    print('Mode: ' + str(mode))

    # Load test data
    with open(testData, 'rb') as file:
        allTestData = pkl.load(file, encoding='latin1')
        testImages = allTestData['images']
        testLabels = allTestData['info']

    # Load pre-trained model
    if modelFile is not None:
        encoder = keras.models.load_model(modelFile)
        with open(keyFile, 'rb') as file:
            key = pkl.load(file, encoding='latin1')
    else:
        modelFileNameTag = fileNameTag + '_test'
        model = makeAndTrainAreaModel(trainData, aeEpochs, classifierEpochs, 
                                      fileNameTag=modelFileNameTag)
        encoder = model['model']
        # The key is the list of reference area values. It is essential to the 
            # interpretation of the model's responses
        key = model['key']

    # Obtain and save model responses
    responses = encoder.predict(testImages)

    # Prepare file to hold model's responses
    outputFile = fileNameTag + '_test_results.txt'
    headers = '{0}\t{1}\t{2}\n'.format('Trial', 'Stimulus', 'ModelResponse')
    with open(outputFile, 'w') as file:
        file.write(headers)
        file.close()

    for response in responses:
        processed_response = ((response > np.mean(response)).astype(int)).tolist()
        for i in range(len(processed_response) + 1):
            # Recall that elements of model's output are zeros and ones
            # Because the activation function of the model's second-to-last layer
                # is softmax, this layer's elements are mutually inhibitive; 
                # thus the network's output is such that the first elements are 
                # ones, and after those, all others are zero
            if processed_response[i] == 1 and processed_response[i+1] == 0:
                # We use the key (reference areas) to identify the largest value 
                    # that the network indicates the image's area exceeds, which 
                    # is the greatest-index element with a value of one
                areaJudgment = key[i]
                break
        # Trial number and stimulus image number
        responseIndex = np.where(responses == response)[0][0]
        trial = math.floor(responseIndex / 2)
        stimulus = (responseIndex % 2) + 1
        # Write model response and trial identification information to file
        contents = '{0}\t{1}\t{2}\n'.format(
                trial, stimulus, areaJudgment)
        with open(outputFile, 'a') as file:
            file.write(contents)
            file.close()