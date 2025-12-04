import time
import numpy as np
import SIFT
from sklearn.ensemble import RandomForestClassifier


class Codebook():

    def __init__(self, n_trees, max_samples, max_features):
        self.n_trees = n_trees
        self.forest = RandomForestClassifier(
            n_jobs= 14,
            n_estimators = self.n_trees,
            max_depth= 3,
            max_features=max_features,
            max_samples=max_samples
                )

    def compute_tree_leaf_indices(self, tree):
        all_leaf_nodes = np.where(tree.tree_.children_left == -1)[0]
         
        idMapping = {}
        counter = 0
        for leaf in all_leaf_nodes:
            idMapping[leaf] = counter
            counter += 1

        return idMapping

    def compute_forest_mappings(self):
        mappings = []

        for i, tree in enumerate(self.forest.estimators_):
            mappings.append(self.compute_tree_leaf_indices(tree))
        
        return mappings


    def fit(self, X, y):
        stime = time.time()
        forest = self.forest

        descriptors, labels = SIFT.getDenseSIFTWithLabels(X, y)
        #reset the train images to gain memory

        forest.fit(descriptors, labels)
        #print('RF algorithm took ', time.time()-stime)

        n_leaves = sum(estimator.get_n_leaves() for estimator in forest.estimators_)

        #we have to create a mapping node_id -> leaves

        print('Our codebook has ', n_leaves, " k")

        self.n_leaves = n_leaves
        self.forest = forest

        self.mappings = self.compute_forest_mappings()


    def getHistogram(self, image):
        #we get the decriptor of the image, and then we 
        descriptors = SIFT.getDenseSIFT([image])

        leaf_indexes = self.forest.apply(descriptors)

        histogram = np.zeros(self.n_leaves)
        offset = 0
        for i, tree in enumerate(self.forest.estimators_):
            leafs = leaf_indexes[:, i]
            temp_histogram = np.zeros(len(self.mappings[i]))
            unique, counts = np.unique(leafs, return_counts=True)
            for node_id, count in zip(unique, counts):
                temp_histogram[self.mappings[i][node_id]] = count
            histogram[offset:offset + len(temp_histogram)] = temp_histogram
            offset += len(temp_histogram)

        return histogram


