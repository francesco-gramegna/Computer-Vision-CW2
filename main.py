import utils
import SIFT



images = utils.loadImages('Caltech_101')

descriptors = SIFT.getDenseSIFT(images)






#utils.getData()


