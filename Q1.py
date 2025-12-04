import numpy as np
import DualPixelTree

import matplotlib.pyplot as plt
import GridSearch
import analysisUtils
import Classifier
import utils
import kMeansCB
import RFCodebook
import SIFT



def main():
    trainX, trainY, testX, testY , classMappings = utils.getData()    

    print(len(trainX), " train images")
    print(len(testX), " test images")
    print(classMappings)

    #descriptor tests
    

    kmeansCB = kMeansCB.Codebook(500, [5, 10], True)
    kmeansCB.fit(trainX, trainY)

    
    kmeansCB2 = kMeansCB.Codebook(500, [5, 10], False)
    kmeansCB2.fit(trainX, trainY)


    
    colors = plt.cm.tab10.colors
    invMappings = {classMappings[i]: i for i in classMappings}

    plt.subplot(1, 2, 1)
    for i in range(len(classMappings)):
        first = True
        for j in trainX[i * 30 : (i+1) * 30]:
            if first:
                plt.plot(kmeansCB.getHistogram(j), color=colors[i], alpha=0.5, label=invMappings[i])
                first = False
            else:
                plt.plot(kmeansCB.getHistogram(j), color=colors[i], alpha=0.5)

    plt.title("Hitograms per class with class normalized SIFT")
    plt.legend()


    plt.subplot(1, 2, 2)
    for i in range(len(classMappings)):
        first = True
        for j in trainX[i * 30 : (i+1) * 30]:
            if first:
                plt.plot(kmeansCB2.getHistogram(j), color=colors[i], alpha=0.5, label=invMappings[i])
                first = False
            else:
                plt.plot(kmeansCB2.getHistogram(j), color=colors[i], alpha=0.5)

    plt.legend()
    plt.title("Hitograms per class with randmoly picked SIFT")


    plt.show()

    
    pred = Classifier.Classifier(kmeansCB,
                                 n_trees=100,
                                 max_samples=20,
                                 min_samples_split=5,
                                 max_features=400,
                                 dp=False,
                                 commutative=False,
                                 )
  
    pred.fit(trainX, trainY)
    pred2 = Classifier.Classifier(kmeansCB2,
                                 n_trees=100,
                                 max_samples=20,
                                 min_samples_split=5,
                                 max_features=400,
                                 dp=False,
                                 commutative=False,
                                  )
  
    pred2.fit(trainX, trainY)
     
                                   

    print("normalized : " , pred.score(testX, testY))
    print("normal : " , pred2.score(testX, testY))
    

if __name__ == "__main__":
    main()
