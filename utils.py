import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def getImage(loc):
    img = mpimg.imread(loc)

    img = np.array(img)
    print(img.shape)

    plt.imshow(img)
    plt.show()



def loadImages(base_folder):
    """
    Loads all images from each subdirectory in `base_folder`.
    Returns a dictionary: {subdir_name: [numpy arrays of images]}.
    """
    images_dict = {}
    
    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path):
            # Load all images in the subdirectory
            images = []
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(subdir_path, file)
                    img_array = np.array(mpimg.imread(img_path))
                    img_array = greyScale(img_array)
                    images.append(img_array)
            if images:
                images_dict[subdir] = images
    
    return images_dict


def getData():
    imgs = loadImages('Caltech_101')

    trainX = []
    trainY = []
    testX = []
    testY = []

    classCounter = 0
    classMapping = {}
    for _class in imgs:
        classMapping[_class] = classCounter
        classCounter += 1
        #15 images are testing, rest is training
        testIndices = np.random.choise(len(imgs[_class]) , 15, replace=False)
        trainIndices = list(set(range(len(imgs[_class]))) - set(testIndices))
        test = imgs[_class][testIndices]
        train = imgs[_class][trainIndices]

        trainX += train
        trainY += [classMapping[_class]] * len(train)

        testX += test
        testY += [classMapping[_class]] * len(test)


    return trainX, trainY, testX, testY, classMapping


def greyScale(img):
    if(len(img.shape) == 3):
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
        return gray

    return img




