from sklearn.model_selection import GridSearchCV
import Classifier


class findBestParam():
    def __init__(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY


    def fit(self):

        """
        param_grid = {
            'cbParams': [
            {'size': 10},
            {'size': 100},
            {'size':500},
            {'size':1000},
            {'size':1500},
            {'size': 2000},
            {'size': 5000}
                ],
            'n_trees':[5,10,50,100,150,200,300,500,1000],
            'max_samples':[25, 40, 60, 80, 100, 130],
            'min_samples_split':[2, 5, 10],
            'max_features':['log2','auto', 10, 20, 50, 100, 150, 250, 500, 750, 1250],
            'splitter':['best', 'random', 'twoPixel']
            }
            """

        param_grid = {
            'cbParams': [{'size':100}],
            'n_trees':[15],
            'max_samples' : [25],
            'min_samples_split': [3],
            'max_features':['log2'],
            'splitter':['best']
                }

        grid = GridSearchCV(
        estimator=Classifier.FullClassifier(),
        param_grid=param_grid,
        verbose=3,  
        n_jobs=-1,
        error_score='raise' 
        )
        grid.fit(self.trainX, self.trainY)






        


