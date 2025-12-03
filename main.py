import numpy as np
import DualPixelTree
import matplotlib.pyplot as plt
import GridSearch
import analysisUtils
import Classifier
import utils
import kMeansCB
import RFCodebook




def main():

    trainX, trainY, testX, testY , classMappings = utils.getData()    

    print(len(trainX), " train images")
    print(len(testX), " test images")
    print(classMappings)

    kmeansCB = kMeansCB.Codebook(100)
    kmeansCB.fit(trainX, trainY)

    #plt.imshow(testX[0])
    #plt.show()

    #plt.plot(kmeansCB.getHistogram(testX[0]))
    #plt.plot(kmeansCB.getHistogram(testX[1]))

    #plt.show()

    pred = Classifier.Classifier(kmeansCB,
                                 n_trees=100,
                                 max_samples=20,
                                 min_samples_split=5,
                                 max_features=400,
                                 dp=False,
                                 commutative=False,
                                 )
    pred2 = Classifier.Classifier(kmeansCB,
                                 n_trees=100,
                                  max_samples=20,
                                 min_samples_split=5,
                                 max_features=400,
                                 dp=DualPixelTree.DPabs,
                                 commutative=True,
                                  )
   
    #pred2Pixel = Classifier.Classifier(kmeansCB, trainX, trainY, twoPixel=True)   

    pred.fit(trainX, trainY)
    pred2.fit(trainX, trainY)
    #pred2Pixel.fit()

    print("normal : " , pred.score(testX, testY))
    print("two pixel : ", pred2.score(testX, testY))

    analysisUtils.plotConfusion(pred, testX,testY, classMappings)
    analysisUtils.plotConfusion(pred2, testX,testY, classMappings)

    """
    #pred = Classifier.FullClassifier()
    #pred.fit(trainX,trainY)
    #Kprint(pred.score(testX,testY))
    
    #finder = GridSearch.findBestParam(trainX, trainY, testX, testY)
    #finder.fit()

    print("\n=== MANUAL TEST ===")
    test_clf = Classifier.FullClassifier(
    cbParams={'size':100},
    n_trees=15,
    max_samples=25,
    min_samples_split=3,
    max_features='log2'
)
    test_clf.fit(trainX, trainY)
    print("Manual test score:", test_clf.score(testX, testY))

    print("\n=== GRIDSEARCH TEST ===")
    finder = GridSearch.findBestParam(trainX, trainY, testX, testY)
    finder.fit()
    """

main()



