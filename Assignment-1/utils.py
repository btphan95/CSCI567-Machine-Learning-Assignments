from typing import List

import numpy as np


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    y_true = np.reshape(y_true,(len(y_true),1))
    mse = (1/len(y_true))*np.sum((y_true-y_pred)**2)
    return mse
    raise NotImplementedError


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    real_labels = np.array(real_labels).reshape((len(real_labels),1))
    predicted_labels = np.array(predicted_labels).reshape((len(predicted_labels),1))
    # print('real',real_labels)
    # print('predicted',predicted_labels)
    #precision is  the number of correct positive results divided by the number of all positive results returned by the classifier
    precision = np.sum((real_labels ==1) & (predicted_labels==1)) / np.sum(predicted_labels)

    #recall is the number of correct positive results divided by the number of all positive results that are in real_labels
    recall = np.sum((real_labels ==1) & (predicted_labels==1)) / np.sum(real_labels)
    # print('precision',precision)
    # print('recall',recall)
    f1 = 2*(precision*recall)/(precision+recall)
    if np.isnan(f1):
        return 0
    else:
        return f1
    raise NotImplementedError


def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:
    features = np.array(features)
    features_extended = np.ones((features.shape[0],features.shape[1]*k))
    for i in range(k):
        features_extended[:,features.shape[1]*i:features.shape[1]*(i+1)] = np.power(np.copy(features),i+1)
    return features_extended.tolist()
    #raise NotImplementedError


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    dist = np.linalg.norm(point1 - point2)
    return dist
    #raise NotImplementedError


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    dist = np.dot(point1,point2)
    return dist
    raise NotImplementedError


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    dist = -np.exp(-0.5*(np.power(np.linalg.norm(point1 - point2),2)))
    return dist
    raise NotImplementedError


def normalize(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[3, 4], [1, -1], [0, 0]],
    the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
    """
    features = np.array(features)
    for i in range(len(features)):
        features[i] = features[i]/np.linalg.norm(features[i])
    return features
    raise NotImplementedError


def min_max_scale(features: List[List[float]]) -> List[List[float]]:
    """
    normalize the feature vector for each sample . For example,
    if the input features = [[2, -1], [-1, 5], [0, 0]],
    the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
    """
    #raise NotImplementedError
