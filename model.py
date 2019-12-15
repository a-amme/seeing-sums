import numpy as np 
import tensorflow as tf 
import keras
from keras.models import Model 
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.layers import UpSampling2D, BatchNormalization
import pickle as pkl
from autoencoder import makeAndTrainModel

num_reference_areas = 20

# Load data
with open('dataset.txt', 'rb') as file:
    data = pkl.load(file, encoding='latin1')['x']

input_shape = data[0].shape

reference_areas = []
total_area = data[0].shape[0] * data[0].shape[1]
step = total_area / num_reference_areas
value = 0
for i in range(num_reference_areas):
    reference_areas.append(value)
    value += step

# Make and train Auto-Encoder
ae = makeAndTrainModel(data, training_epochs=2)

# Separate Encoder portion and add classifier
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
outputs = Dense(num_reference_areas, activation='softmax', use_bias=False)(flatten)

encoder = Model(input=inputs, output=outputs)
encoder.compile('adam', loss='binary_crossentropy')

encoder_weights = ae.get_weights()[:10] + [encoder.get_weights()[-1]]
encoder.set_weights(encoder_weights)

encoder.summary()

# Train classifier
# Prepare label vectors
""" To train the classifier to represent the model's area jugments as best as 
    possible, we will assign each of its nodes to one of the area reference
    values and reward the network when it sees an image with area greater than 
    or equal to this value and turns the node on, and punish it when it fails 
    to do so. In order to determine whether the network should be rewarded or 
    penalized, we need the "right answer"––that is, the area of each training 
    image represented as a vector of 1s and 0s corresponding to each of the 
    reference area values. 
"""

# Process data to extract area (i.e. total "on" pixels) information
# Convert 3-channel data to single-channel
reduced_data = data.dot([1, 1, 1])
# Generate area value for each image
new_shape = reduced_data.shape[0], reduced_data.shape[1] * reduced_data.shape[2]
flattened_data = np.reshape(reduced_data, new_shape) / 3
area_data = np.sum(flattened_data, axis=-1)
# Generate label vectors by comparing area values to reference values
y = np.greater(area_data, reference_areas[0])
y = np.reshape(y, (y.shape[0], 1))
for reference in reference_areas[1:]:
    comparison = np.greater(area_data, reference)
    comparison = np.reshape(comparison, (comparison.shape[0], 1))
    y = np.concatenate((y, comparison), axis=1)

# Train classifier
encoder.fit(data, y, epochs=50, verbose=1)