import pickle as pkl
import numpy as np
import os

def concatenate(outputFile, dataFiles, cleanup=False):

    """
    Combine several data files into one.
        outputFile: desired output filename as a string
        dataFiles: list of files to be concatenated as strings
        cleanup: whether to delete the data files after consolidation
    """

    with open(dataFiles[0], 'rb') as file:
        data = pkl.load(file)
        x = data['x']
        aa = data['aa']

    for filename in dataFiles[1:]:
        with open(filename, 'rb') as file:
            add_data = pkl.load(file)
            add_x = add_data['x']
            add_aa = add_data['aa']
            assert x[0].shape == add_x[0].shape, "Images must be same size"
            x = np.concatenate((x, add_x))
            aa = np.concatenate((aa, add_aa))

    dataDict = {
        'x': x,
        'aa': aa
    }

    with open(outputFile, 'wb') as file:
        pkl.dump(dataDict, file)

    print("New dataset contains " + str(x.shape[0]) + " items")

    if cleanup:
        for filename in dataFiles:
            os.remove(filename)