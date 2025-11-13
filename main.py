import matplotlib.pyplot as plt
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

    kmeansCB = kMeansCB.Codebook(trainX, 1750)
    kmeansCB.fit()

    #plt.imshow(testX[0])
    #plt.show()

    #plt.plot(kmeansCB.getHistogram(testX[0]))
    #plt.plot(kmeansCB.getHistogram(testX[1]))

    #plt.show()

    pred = Classifier.Classifier(kmeansCB, trainX, trainY)   
    pred2Pixel = Classifier.Classifier(kmeansCB, trainX, trainY, twoPixel=True)   

    pred.fit()
    pred2Pixel.fit()

    print("normal : " , pred.accuracy(testX, testY))
    print("two pixel : ", pred2Pixel.accuracy(testX, testY))

    analysisUtils.plotConfusion(pred2Pixel, testX,testY, classMappings)

main()



