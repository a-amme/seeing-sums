import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.layers import UpSampling2D, BatchNormalization
import pickle as pkl
from autoencoder import makeAndTrainModel

def makeAndTrainAreaModel(dataset,
                          classifier_training_epochs, 
                          ma,
                          aa, 
                          fileNameTag=None, 
                          ae_file=None, 
                          ae_training_epochs=None,
                          num_reference_areas=20):
    
    """
    All in the name.
    PARAMETERS:
        dataset: str or dictionary of numpy arrays; filename of dataset or dataset
        classifier_training_epochs: int, number of epochs to train classifier for
        ma: bool, whether to train an MA model
        aa: bool, whether to train an AA model
        fileNameTag: str, name to base model and key filenames off of; if one 
            is provided, model will be saved 
        ae_file: str; filename of a saved autoencoder if one is being used
        ae_training_epochs: int, number of epochs to train AE for. Only 
            required if no ae_file given.
        num_reference_areas: int, number of nodes in the classifier which will 
            be added to the autoencoder. Each node will correspond to one 
            'reference' area value
    """

    # Load data
    if isinstance(dataset, str):
        with open(dataset, 'rb') as file:
            allData = pkl.load(file, encoding='latin1')
            data = allData['x']
            aa_labels = allData['aa']
    else:
        data = dataset['x']
        aa_labels = dataset['aa']
    
    # Process data to extract area (i.e. total "on" pixels) information
    # Convert 3-channel data to single-channel
    reduced_data = data.dot([1, 1, 1])
    # Generate area value for each image
    new_shape = reduced_data.shape[0], reduced_data.shape[1] * reduced_data.shape[2]
    flattened_data = np.reshape(reduced_data, new_shape) / 3
    area_data = (data.shape[1] * data.shape[2]) - np.sum(flattened_data, axis=-1)

    input_shape = data[0].shape

    if ma:
        ma_reference_areas = []
        total_area = area_data.max()
        step = total_area / num_reference_areas
        value = 0
        for i in range(num_reference_areas):
            ma_reference_areas.append(value)
            value += step
    if aa:
        aa_reference_areas = []
        total_area = aa_labels.max() 
        step = total_area / num_reference_areas
        value = 0
        for i in range(num_reference_areas):
            aa_reference_areas.append(value)
            value += step

    # Make and train Auto-Encoder
    if ae_file is not None:
        ae = load_model(ae_file)
    else: 
        ae = makeAndTrainModel(data, training_epochs=ae_training_epochs)

    # Separate Encoder portion and add classifier
    # MA
    if ma:
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
        outputs = Dense(num_reference_areas, activation='softmax', use_bias=True)(flatten)

        ma_model = Model(input=inputs, output=outputs)
        ma_model.compile('adam', loss='binary_crossentropy')

        ma_model_weights = ae.get_weights()[:10] + ma_model.get_weights()[-2:]
        ma_model.set_weights(ma_model_weights)

        ma_model.summary()

    # AA
    if aa: 
        aa_model = Model(input=inputs, output=outputs)
        aa_model.compile('adam', loss='binary_crossentropy')

        aa_model_weights = ae.get_weights()[:10] + aa_model.get_weights()[-2:]
        aa_model.set_weights(aa_model_weights)

        aa_model.summary()

    # Prepare label vectors
    # MA
    """ To train the classifier to represent the model's MA jugments as best as 
        possible, we will assign each of its nodes to one of the area reference
        values and reward the network when it sees an image with area greater than 
        or equal to this value and turns the node on, and punish it when it fails 
        to do so. In order to determine whether the network should be rewarded or 
        penalized, we need the "right answer"––that is, the area of each training 
        image represented as a vector of 1s and 0s corresponding to each of the 
        reference area values. 
    """
    if ma: 
        ma_y = np.greater(area_data, ma_reference_areas[0])
        ma_y = np.reshape(ma_y, (ma_y.shape[0], 1))
        for reference in ma_reference_areas[1:]:
            comparison = np.greater(area_data, reference)
            comparison = np.reshape(comparison, (comparison.shape[0], 1))
            ma_y = np.concatenate((ma_y, comparison), axis=1)
        ma_y = ma_y.astype(int)

    # AA
    if aa: 
        aa_labels = np.array(aa_labels)
        aa_y = np.greater(aa_labels, aa_reference_areas[0])
        aa_y = np.reshape(aa_y, (aa_y.shape[0], 1))
        for reference in aa_reference_areas[1:]:
            comparison = np.greater(aa_labels, reference)
            comparison = np.reshape(comparison, (comparison.shape[0], 1))
            aa_y = np.concatenate((aa_y, comparison), axis=1)
        aa_y = aa_y.astype(int)
        print(aa_y.shape)

    # Train MA model
    if ma: 
        ma_model.fit(data, ma_y, epochs=classifier_training_epochs, verbose=1)
    # Train AA model
    if aa: 
        aa_model.fit(data, aa_y, epochs=classifier_training_epochs, verbose=1)

    if fileNameTag is not None:
        keyDict = {}
        returnDict = {}
        if ma: 
            MAModelFile = fileNameTag + '_ma_model.h5'
            ma_model.save(MAModelFile)
            keyDict['MA'] = ma_reference_areas
            returnDict['MA key'] = ma_reference_areas
            returnDict['MA model'] = ma_model
        if aa:
            AAModelFile = fileNameTag + '_aa_model.h5'
            aa_model.save(AAModelFile)
            keyDict['AA'] = aa_reference_areas
            returnDict['AA key'] = aa_reference_areas
            returnDict['AA model'] = aa_model
        keyFile = fileNameTag + '_keys.txt'
        with open(keyFile, 'wb') as file:
            pkl.dump(keyDict, file)
    
    return(returnDict)