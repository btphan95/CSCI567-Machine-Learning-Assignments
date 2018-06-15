from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        self.features = np.array(features)
        self.labels = np.array(labels)
        #raise NotImplementedError

    def predict(self, features: List[List[float]]) -> List[int]:
        #array of predicted values
        features = np.array(features)
        # print(features.shape)
        #array of distance values
        distances = np.zeros((len(self.features),1))
        print('k',self.k)
        #array of the class values for the indices of the k smallest distance values
        class_vals = np.zeros((self.k))
        preds = np.zeros((len(features),1))
        for i in range(len(preds)):
            sample = features[i]
            for j in range(len(distances)):
                # if j < 3:
                #     #print('dot',np.dot(sample,self.features[j]))

                distances[j] =  self.distance_function(sample, self.features[j])
            min_dists = np.argsort(distances,axis=0)[:self.k]
            class_vals = self.labels[min_dists]
            # if i < 3:
            # # #     print('distances',distances)
            #       print('min_dists' , min_dists)
            #      # print('argmin',np.min(distances))
            #       print('class vals', class_vals)
            # print('mode',scipy.stats.mode(class_vals))
            preds[i] = scipy.stats.mode(class_vals)[0][0]
        # print(preds)

        return preds
        raise NotImplementedError


if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
