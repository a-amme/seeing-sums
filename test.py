import numpy as np 
import tensorflow as tf 
import keras
import pickle as pkl
import math
from model import makeAndTrainAreaModel
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape

# test('consoliData.txt', 'test_data.txt', 1, 1)

def discriminations(fileNameTag, testData, trainData=None, aeEpochs=None, 
                    classifierEpochs=None, MAModelFile=None, AAModelFile=None, 
                    keyFile=None):

    """
    Present a set of images to the model and record its area judgments.
    PARAMETERS:
    Note that the function will throw an error unless all of trainData, 
    aeEpochs, and classifierEpochs or all of modelFile and keyFile are provided.
    If MAModelFile, AAModelFile, and keyFile are given, the test will proceed 
    using a saved model. Otherwise, if trainData, aeEpochs, and classifierEpochs 
    are given, the test function will create a new model.
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
        MAModelFile: str, the name of a file containing a saved model
        AAModelFile: str, the name of a file containing a saved model
        keyFile: str, the name of a file containing the key corresponding to the 
            saved models at MAModelFile and AAModelFile
    RETURNS:
        none.
    """

    # Ensure arguments passed to function allow either for the use of a pre-trained 
        # model or the generation of a new one
    mode = None
    if None not in [MAModelFile, keyFile] or None not in [AAModelFile, keyFile]:
        mode = 'pre-trained model'
    elif None not in [trainData, aeEpochs, classifierEpochs]:
        mode = 'fresh model'
    assert mode != None, 'Insufficient information provided. Check arguments to function call.'
    print('Mode: ' + str(mode))

    # Load test data
    with open(testData, 'rb') as file:
        allTestData = pkl.load(file, encoding='latin1')
        testImages = allTestData['images']

    ma_model = None
    aa_model = None

    # Load pre-trained model
    if MAModelFile is not None:
        ma_model = keras.models.load_model(MAModelFile)
        with open(keyFile, 'rb') as file:
            keys = pkl.load(file, encoding='latin1')
            ma_key = keys['MA']
    if AAModelFile is not None:
        aa_model = keras.models.load_model(AAModelFile)
        with open(keyFile, 'rb') as file:
            keys = pkl.load(file, encoding='latin1')
            ma_key = keys['AA']
    if None not in [MAModelFile, AAModelFile]:
        modelFileNameTag = fileNameTag + '_area_judgments'
        model = makeAndTrainAreaModel(trainData, aeEpochs, classifierEpochs, 
                                      fileNameTag=modelFileNameTag)
        ma_model = model['MA model']
        aa_model = model['AA model']
        # The key is the list of reference area values. It is essential to the 
            # interpretation of the model's responses
        ma_key = model['MA key']
        aa_key = model['AA key']

    MA = False
    AA = False
    if ma_model is not None:
        MA = True
    if aa_model is not None:
        AA = True

    # Obtain and save model responses
    if MA:
        ma_responses = ma_model.predict(testImages)
    if AA:
        aa_responses = aa_model.predict(testImages)

    # Prepare file to hold model's responses
    outputFile = fileNameTag + '_test_results.txt'
    if MA and AA:
        headers = '{0}\t{1}\t{2}\t{3}\n'.format('Trial', 'Stimulus', 
                                                'ModelResponseMA', 
                                                'ModelResponseAA')
    elif MA:
        headers = '{0}\t{1}\t{2}\n'.format('Trial', 'Stimulus', 'ModelResponseMA')
    elif AA: 
        headers = '{0}\t{1}\t{2}\n'.format('Trial', 'Stimulus', 'ModelResponseAA')
    with open(outputFile, 'w') as file:
        file.write(headers)
        file.close()

    # The models' responses are vectors of zeros and ones. We must convert them 
        # into area judgments using the key of reference area values. 
    for i in range(testImages.shape[0]):
        if MA:
            ma_response = ma_responses[i]
            processed_ma_response = ((ma_response > np.mean(ma_response)).astype(int)).tolist()
            for j in range(len(processed_ma_response) + 1):
                # Recall that elements of model's output are zeros and ones
                # Because the activation function of the model's second-to-last layer
                    # is softmax, this layer's elements are mutually inhibitive; 
                    # thus the network's output is such that the first elements are 
                    # ones, and after those, all others are zero
                if processed_ma_response[j] == 1 and processed_ma_response[j+1] == 0:
                    # We use the key (reference areas) to identify the largest value 
                        # that the network indicates the image's area exceeds, which 
                        # is the greatest-index element with a value of one
                    MAJudgment = ma_key[j]
                    break
        if AA:
            aa_response = aa_responses[i]
            processed_aa_response = ((aa_response > np.mean(aa_response)).astype(int)).tolist()
            for j in range(len(processed_aa_response) + 1):
                if processed_aa_response[j] == 1 and processed_aa_response[j+1] == 0:
                    AAJudgment = aa_key[j]
                    break
        # Trial number and stimulus image number
        trial = math.floor(i / 2)
        stimulus = (i % 2) + 1
        # Write model response and trial identification information to file
        if MA and AA:
            contents = '{0}\t{1}\t{2}\t{3}\n'.format(trial, stimulus, MAJudgment, AAJudgment)
        elif MA:
            contents = '{0}\t{1}\t{2}\n'.format(trial, stimulus, MAJudgment)
        elif AA:
            contents = '{0}\t{1}\t{2}\n'.format(trial, stimulus, AAJudgment)

        with open(outputFile, 'a') as file:
            file.write(contents)
            file.close()


def representationsPairs(fileNameTag, testData, autoencoderFile): 

    """
    Present a set of paired images to the model and record the euclidean 
    distance between its representations. 
    PARAMETERS:
        fileNameTag: str from which the name of the results file will be 
            generated
        testData: str; name of file in which test images are stored
        autoencoderFile: str; name of file to which trained autoencoder is saved
    RETURNS:
        none.
    """

    # Load test images
    with open(testData, 'rb') as file:
        data = pkl.load(file, encoding='latin1')
        images = data['images']
    input_shape = images[0].shape

    # Load autoencoder
    ae = load_model(autoencoderFile)

    # Isolate encoder
    inputs = Input(input_shape)
    x_1 = Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False)(inputs)
    x_2 = MaxPooling2D((2,2), padding='same', trainable=False)(x_1)
    x_3 = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(x_2)
    x_4 = MaxPooling2D((2, 2), padding='same', trainable=False)(x_3)
    x_5 = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(x_4)
    x_6 = MaxPooling2D((2, 2), padding='same', trainable=False)(x_5)
    x_7 = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(x_6)
    x_8 = MaxPooling2D((2, 2), padding='same', trainable=False)(x_7)
    x_9 = Conv2D(32, (3, 3), activation='relu', padding='same', trainable=False)(x_8)
    encoded = MaxPooling2D((2, 2), padding='same', trainable=False)(x_9)
    flatten = Flatten()(encoded)

    encoder = Model(input=inputs, output=flatten)
    encoder.compile('adam', loss='binary_crossentropy')

    weights = ae.get_weights()[:10]
    encoder.set_weights(weights)

    # Perform test
    representations = encoder.predict(images)

    # Initialize output file
    outputFileName = fileNameTag + '_representation_distance_paired.txt'
    headers = '{0}\t{1}\n'.format('Trial', 'Distance')
    with open(outputFileName, 'w') as file:
        file.write(headers)
        file.close()

    # Compute euclidean distances and write result to output file
    for trial in range(0, images.shape[0], 2):
        imageOneRep = representations[trial]
        imageTwoRep = representations[trial + 1]
        distance = np.linalg.norm(imageOneRep - imageTwoRep)
        data = '{0}\t{1}\n'.format(str(trial / 2), str(distance))
        with open(outputFileName, 'a') as file:
            file.write(data)
            file.close()