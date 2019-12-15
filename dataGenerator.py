import numpy as np
from PIL import Image
from psychopy import visual, core, data, event, logging, gui
import math
import os
import pickle as pkl


def generateCircles(numerosity, minRad, maxRad, padding, size):

    """
    Generate radius and positions for a single image
    ARGUMENTS
        numerosity: int representing number of objects to generate
        minRad: float representing minimum acceptable radius
        maxRad: float representing maximum acceptable radius
        padding: float representing minimum desired space between objects
        size: float representing desired image side length
    RETURNS
        array with numerosity elements of the form [radius, (x, y)]
    """

    loop = True
    while loop == True:
        loop = False
        dots = []
        counter = 0
        while len(dots) < numerosity: 
            counter += 1
            dotAttempt = np.random.uniform(minDotRad, maxDotRad)
            posBound = (size / 4) - (dotAttempt + padding)
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
            if (counter - numerosity) > 8:
                loop = True
                break
    return dots


def generateRectangles(numerosity, minSide, maxSide, padding, size):

    """
    Generate radius and positions for a single image
    ARGUMENTS
        numerosity: int representing number of objects to generate
        minSide: float representing minimum acceptable side length
        maxSide: float representing maximum acceptable side length
        padding: float representing minimum desired space between objects
        size: float representing desired image side length
    RETURNS
        array with numerosity elements of the form [side, (x, y)]
    """

    loop = True
    while loop == True:
        loop = False
        shapes = []
        counter = 0
        while len(shapes) < numerosity: 
            counter += 1
            shapeAttempt = np.random.uniform(minSide, maxSide)
            posBound = (size / 4) - (shapeAttempt + padding)
            posAttempt = np.random.uniform(-posBound, posBound), np.random.uniform(-posBound, posBound)
            # Check whether this dot fits the others
            goodShape = False
            if len(shapes) == 0:
                goodShape = True
            else:
                goodShape = True
                for shape in shapes:
                    centerDistanceY = max(shape[1][1], posAttempt[1]) -  min(shape[1][1], posAttempt[1])
                    centerDistanceX = max(shape[1][0], posAttempt[0]) -  min(shape[1][0], posAttempt[0])
                    minimumDistanceX = shape[0] / 2 + shapeAttempt / 2 + padding
                    minimumDistanceY = shape[0] / 2 + shapeAttempt / 2 + padding
                    if centerDistanceX < minimumDistanceX and centerDistanceY < minimumDistanceY:
                        goodShape = False
            if goodShape: 
                shapes.append([shapeAttempt, posAttempt])
            if (counter - numerosity) > 8:
                loop = True
                break
    return shapes


num = 2500 # this is an upper bound
low = 1
high = 13
imagesPerNumerosity = num / (high - low + 1)
size = 64
padding = 1
shape = 'rectangle'

tag = 'rect'

# Generate num two-color stimuli with numerosities 
#   between low and high

images = np.empty((num, size, size, 3))
labels = np.empty(num, dtype='int')

for numerosity in range(low, high + 1):
    for n in range(imagesPerNumerosity):
        if shape == 'circle': 
            maxDotArea = (size - (2 * padding))**2 /  (8 * numerosity)
            minDotArea = maxDotArea / 10
            maxDotRad = math.sqrt(maxDotArea / math.pi)
            minDotRad = math.sqrt(minDotArea / math.pi)
            dots = generateCircles(numerosity, minDotRad, maxDotRad, padding, size)
                    # Draw image with psychopy
            win = visual.Window(size=(size / 2, size / 2), units='pix', fullscr=False, screen=0, monitor='testMonitor', color='#ffffff', colorSpace='rgb')
            win.flip()
            for dot in dots:
                circle = visual.Circle(win, units='pix', radius=dot[0], pos=dot[1], fillColor='#000000', lineWidth=0.0)
                circle.draw()
        if shape == 'rectangle':
            maxSide = size / numerosity
            minSide = 3
            rectangles = generateRectangles(numerosity, minSide, maxSide, padding, size)
            win = visual.Window(size=(size / 2, size / 2), units='pix', fullscr=False, screen=0, monitor='testMonitor', color='#ffffff', colorSpace='rgb')
            win.flip()
            for rect in rectangles:
                rectangle = visual.Rect(win, units='pix', width=rect[0], height=rect[0], pos=rect[1], fillColor='#000000', lineWidth=0.0)
                rectangle.draw()
        win.flip()
        win.getMovieFrame(buffer='front')
        imageName = "pngs/" + tag + "_" + str(numerosity) + "_" + str(n + 1) + ".png"
        win.saveMovieFrames(imageName)
        win.close()
        # Convert image to array
        image = Image.open( imageName  )
        # Normalize image
        imageAsArray = np.array(  image ) / 255.0
        # Add images to dataset array
        imageIndex = (range(low, high + 1).index(numerosity) * imagesPerNumerosity) + n
        images[imageIndex] = imageAsArray
        labels[imageIndex] = numerosity
        # Delete png (save for one of each numerosity, as a sanity check)
        if n != 0:
            os.remove(imageName)
dataDict = {
    "x": images,
    "y": labels
}

filename = tag + '_dot_displays.txt'
with open(filename, 'wb') as file:
    pkl.dump(dataDict, file)