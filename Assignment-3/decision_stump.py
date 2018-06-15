import numpy as np
from typing import List
from classifier import Classifier

class DecisionStump(Classifier):
    def __init__(self, s:int, b:float, d:int):
        self.clf_name = "Decision_stump"
        self.s = s
        self.b = b
        self.d = d

    def train(self, features: List[List[float]], labels: List[int]):
        pass
        
    def predict(self, features: List[List[float]]) -> List[int]:
        ##################################################
        # TODO: implement "predict"
        ##################################################
        features = np.array(features)
        # print('features',features.shape)
        # print('features[i] in stump',len(features))
        feature = features[:,self.d]
        feature = feature > self.b
        feature = feature * 2
        feature = feature - 1
        feature = feature * self.s
        # print('type',type(feature[0]))
        return feature.tolist()