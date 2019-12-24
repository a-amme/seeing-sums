import numpy as np
from PIL import Image
from psychopy import visual, core, data, event, logging, gui
import math
import os
import pickle as pkl


def generateTests(size, padding, instances, ratios, tag, type='all'):

    """ 
    Name captures function.
    PARAMETERS:
        size: int representing desired side length of square images
        padding: int or float representing desired minimum number of pixels 
            between dots or dots and the image edges
        instances: int representing the desired number of unique instances per 
            variable value combination
        ratios: list of floats representing desired area ratios
        tag: str to occur at start of names of all files saved
        type: 'all' (default), 'aa control', 'ma control', or 'control', 
            indicating which type of trials to generate, corresponding to all 
            kinds, only trials in which paired stimuli have the same AA and 
            differing MA, only trials where paired images have the same MA and 
            differing AA, or only trials in which both MA and AA are equated
    """
    
    # Make a list containing a dictionary representing each pair of images to 
        # be generated
    trials = []
    if type in ['all', 'ma control']:
        for AARatio in ratios: 
            for numerosityCongruence in ['congruent', 'incongruent']:
                for trial in range(instances):
                    trials.append(
                        {
                            'AA Ratio': AARatio,
                            'MA Ratio': 1.0,
                            'numerosity congruence': numerosityCongruence,
                            'instance ID': trial
                        }
                    )
    if type in ['all', 'aa control']:
        for MARatio in ratios: 
            for numerosityCongruence in ['congruent', 'incongruent']:
                for trial in range(instances):
                    trials.append(
                        {
                            'AA Ratio': 1.0,
                            'MA Ratio': MARatio,
                            'numerosity congruence': numerosityCongruence,
                            'instance ID': trial
                        }
                    )
    if type in ['all', 'control']:
        for trial in range(instances):
            trials.append(
                {
                    'AA Ratio': 1.0,
                    'MA Ratio': 1.0,
                    'numerosity congruence': 'congruent',
                    'instance ID': trial
                }
            )

    # Set up materials to be saved upon conclusion of generating
    infoFile = 'Stimuli/' + tag + '_trial_info'
    headers = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n'.format(
              'TrialNumber', 'InstanceID', 'AARatio', 'MARatio', 'lowerAA',
              'higherAA', 'lowerMA', 'higherMA', 'comparisonNumerosity', 
              'numerosityCongruence', 'FileOne', 'FileTwo')
    with open(infoFile, 'wb') as file:
        file.write(headers)
        file.close()
    images = np.empty((len(trials) * 2, size, size, 3))

    for trial in trials:
        # Extract variable values from dictionary
        aaRatio = trial['AA Ratio']
        maRatio = trial['MA Ratio']
        numerosityCongruence = trial['numerosity congruence']
        instance = trial['instance ID']
        trialNumber = trials.index(trial)

        # Setting possible diameter range to correspond with stimuli in Yousif 
            # & Keil, 2019, which had a diameter range of [20, 100] and size of 
            # 400 x 400
        minDiameter = (20. / 400.) * size
        maxDiameter = (100. / 400.) * size
        print(minDiameter, maxDiameter)

        # Generate the intial set of seven dots
        loop = True
        while loop == True:
            loop = False
            dots = []
            aa = 0
            ma = 0
            counter = 0
            while len(dots) < 7: 
                counter += 1
                dotAttempt = np.random.uniform(minDiameter, maxDiameter)
                posBound = (size / 2) - (dotAttempt + padding)
                posAttempt = np.random.uniform(-posBound, posBound), np.random.uniform(-posBound, posBound)
                # Check whether this dot fits the others
                goodDot = False
                if len(dots) == 0:
                    goodDot = True
                else:
                    goodDot = True
                    for dot in dots:
                        centerDistance = math.sqrt((dot[1][0] - posAttempt[0])**2 + 
                            (dot[1][1] - posAttempt[1])**2)
                        if centerDistance < (dot[0] + dotAttempt + padding):
                            goodDot = False
                if goodDot: 
                    dots.append([dotAttempt, posAttempt])
                    aa += 2 * dotAttempt
                    ma += math.pi * dotAttempt**2
                if (counter - 7) > 25:
                    loop = True
                    break
        setOne = {
            'dots': dots,
            'aa': aa,
            'ma': ma
        }
        
        print('set one complete')
        # The second image generated will always have a greater numerosity; 
            # here, we vary whether it has more or less area
        if numerosityCongruence == 'congruent': # More area, more number
            targetAA = setOne['aa'] * aaRatio
            targetMA = setOne['ma'] * maRatio
        else: # Incongruent: less area, more number
            targetAA = setOne['aa'] / aaRatio
            targetMA = setOne['ma'] / maRatio

        numerosity = np.random.uniform(8, 13)

        # Generate the next set of dots
        satisfactory = False
        while satisfactory == False:
            loop = True
            while loop == True:
                loop = False
                dots = []
                aa = 0
                ma = 0
                counter = 0
                while len(dots) < numerosity: 
                    counter += 1
                    dotAttempt = np.random.uniform(minDiameter, maxDiameter)
                    posBound = (size / 2) - (dotAttempt + padding)
                    posAttempt = np.random.uniform(-posBound, posBound), np.random.uniform(-posBound, posBound)
                    # Check whether this dot fits the others
                    goodDot = False
                    if len(dots) == 0:
                        goodDot = True
                    else:
                        goodDot = True
                        for dot in dots:
                            centerDistance = math.sqrt((dot[1][0] - posAttempt[0])**2 + 
                                (dot[1][1] - posAttempt[1])**2)
                            if centerDistance < (dot[0] + dotAttempt + padding):
                                goodDot = False
                    if goodDot: 
                        dots.append([dotAttempt, posAttempt])
                        aa += 2 * dotAttempt
                        ma += math.pi * dotAttempt**2
                    if (counter - numerosity) > 200:
                        loop = True
                        break
            # Accept the second set of dots if its area values differ from the 
                # targets by no more than 5%
            if abs((aa / targetAA) - 1) <= 0.05 and abs((ma / targetMA) - 1) <= 0.05:
                satisfactory == True

        setTwo = {
            'dots': dots,
            'aa': aa,
            'ma': ma,
        }

        print("Rendering...")
        # Render and save images
        fileOne = 'Stimuli/' + tag + str(trialNumber) + "_1"
        fileTwo = 'Stimuli/' + tag + str(trialNumber) + "_2"

        for image in [setOne, setTwo]:
            win = visual.Window(size=(size / 2, size / 2), units='pix', 
                                fullscr=False, screen=0, monitor='testMonitor', 
                                color='#ffffff', colorSpace='rgb')
            win.flip()
            for dot in image['dots']:
                circle = visual.Circle(win, units='pix', radius=dot[0], pos=dot[1], 
                                    fillColor='#000000', lineWidth=0.0)
                circle.draw()
            win.flip()
            win.getMovieFrame(buffer='front')
            if image == setOne:
                imageName = fileOne
                imageIndex = 2 * trialNumber
            else: 
                imageName = fileTwo
                imageIndex = 2 * trialNumber + 1
            win.saveMovieFrames(imageName)
            win.close()
            # Convert image to array
            image = Image.open( imageName  )
            # Normalize image
            imageAsArray = np.array(  image ) / 255.0
            # Add images to dataset array
            images[imageIndex] = imageAsArray

        # Push trial info to list
        """ FORMAT OF TRIALINFO LIST:
        trialInfo = [['TrialNumber', 'InstanceID', 'AARatio', 'MARatio', 'lowerAA',
                'higherAA', 'lowerMA', 'higherMA', 'comparisonNumerosity', 
                'numerosityCongruence', 'FileOne', 'FileTwo']]
        """
        info = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n'.format(
                trialNumber, 
                instance,
                aaRatio,
                maRatio,
                min(setOne['aa'], setTwo['aa']),
                max(setOne['aa'], setTwo['aa']),
                min(setOne['ma'], setTwo['ma']),
                max(setOne['ma'], setTwo['ma']),
                numerosity, 
                numerosityCongruence,
                fileOne,
                fileTwo
                )
        with open(infoFile, 'wb') as file:
            file.write(info)
            file.close()

    # Pickle image data
    imageDict = {'images': images}
    filename = 'test_data_' + tag + '.txt'
    with open('wb') as file:
        pkl.dump(imageDict, file)