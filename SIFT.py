from cyvlfeat.sift import dsift
import random
import numpy as np


def getDenseSIFT(images):
    step=8
    size=8 #todo change for 'multi scale' ...
    all_descr = []

    for _class in images:
        for img in images[_class]:
            frames, descriptors = dsift(img, step=step, size=size, fast=True)
            all_descr.append(descriptors)

    all_descr = np.vstack(all_descr)


    print("Total descriptors : " ,all_descr.shape)

    #now we randomly sample 100K

    indices = np.random.choice(all_descr.shape[0], 100000, replace=False) 

    return all_descr[indices]



