import sys
sys.path.append("/home/francesco/.local/lib/python3.13/site-packages") #For the cyvlfeat 
from cyvlfeat.sift import dsift
import random
import numpy as np


def getDenseSIFT(images):
    #set random seed 
    np.random.seed(13)

    step=8
    size=8 #todo change for 'multi scale' ...
    all_descr = []
    all_frames = []

    for img in images:
        frames, descriptors = dsift(img, step=step, size=size, fast=True)
        all_descr.append(descriptors)

    all_descr = np.vstack(all_descr)


    #print("Total descriptors : " ,all_descr.shape)

    #now we randomly sample 100K, if there are more than 100k
    if(all_descr.shape[0] < 100000):
        return all_descr

    indices = np.random.choice(all_descr.shape[0], 100000, replace=False) 

    return all_descr[indices]


def getDenseSIFTWithLabels(images, classes):
    #set random seed 
    np.random.seed(13)

    step=8
    size=8 #todo change for 'multi scale' ...
    all_descr = []
    all_classes = []

    for img, _class in zip(images, classes):
        frames, descriptors = dsift(img, step=step, size=size, fast=True)
        all_descr.append(descriptors)
        all_classes.append([_class] * len(descriptors))

    all_descr = np.vstack(all_descr)
    all_classes = np.concatenate(all_classes)


    #print("Total descriptors : " ,all_descr.shape)

    #now we randomly sample 100K, if there are more than 100k
    if(all_descr.shape[0] < 100000):
        return all_descr

    indices = np.random.choice(all_descr.shape[0], 100000, replace=False) 

    return all_descr[indices], all_classes[indices]



def getDenseSIFTByImage(img, sizes=[8, 9]):


    imgFrames = []
    imgDesc = []

    for size in sizes:
        step = size
        f, d = dsift(img, step=step, size=size, fast=True)
        size_column = np.full((f.shape[0], 1), size)
        f = np.hstack([f, size_column])

        imgFrames.append(f)
        imgDesc.append(d)

    imgFrames = np.vstack(imgFrames)
    imgDesc = np.vstack(imgDesc)

    return imgFrames, imgDesc



def getDenseSIFTByImages(images, classes, sizes=[8, 9]):

    maxDescr = 100000
    perImageSelection = maxDescr // len(images)
    print("Will get", perImageSelection, "descriptors per image")

    descriptors = np.zeros((len(images), perImageSelection, 128))
    out_frames = np.zeros((len(images), perImageSelection, 3))

    for i, (img, _class) in enumerate(zip(images, classes)):
        imgFrames = []
        imgDesc = []

        for size in sizes:
            step = size
            f, d = dsift(img, step=step, size=size, fast=True)
            size_column = np.full((f.shape[0], 1), size)
            f = np.hstack([f, size_column])

            imgFrames.append(f)
            imgDesc.append(d)

        imgFrames = np.vstack(imgFrames)
        imgDesc = np.vstack(imgDesc)

        if len(imgDesc) < perImageSelection:
            idx = np.random.choice(len(imgDesc), perImageSelection, replace=True)
        else:
            idx = np.random.choice(len(imgDesc), perImageSelection, replace=False)
        descriptors[i] = imgDesc[idx]
        out_frames[i] = imgFrames[idx]

    return out_frames, descriptors

