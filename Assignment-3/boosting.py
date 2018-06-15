import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		features = np.array(features)
		N = features.shape[0]
		h = np.zeros((N))
		# print('features in boosting',len(features))
		for i in range(self.T):
			h = h + np.array(self.clfs_picked[i].predict(features))*self.betas[i]
		h = np.sign(h)
		return h.tolist()

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		
		N = len(features)
		w = np.ones(N)* (1/N)
		for i in range(self.T):
			errors = []
			for i in range(len(list(self.clfs))):
				errors.append(np.dot(w,np.not_equal(labels,list(self.clfs)[i].predict(features))))
			min_error = min(errors)
			h_t = errors.index(min_error)
			preds = [int(x) for x in np.not_equal(labels,list(self.clfs)[h_t].predict(features))]
			preds2 = [x if x==1 else -1 for x in preds]
			self.clfs_picked.append(list(self.clfs)[h_t])
			beta = 0.5*np.log((1-min_error)/min_error)
			self.betas.append(beta)

			w = np.multiply(w, np.exp([x*beta for x in preds2]))
			w = w / np.sum(w)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		f = 0
		pi = np.ones(N)*0.5
		for i in range(self.T):
			#compute the weights
			w = np.multiply(pi,1-pi)
			#compute the working response

			# print('numerator',len(np.divide(np.array(labels)+1,2-pi)))
			# print('w',w.shape)
			z = np.divide((np.array(labels)+1/2)-pi, w)
			errors = []
			for i in range(len(list(self.clfs))):
				errors.append(np.dot(w,np.square(np.subtract(z,list(self.clfs)[i].predict(features)))))
			min_error = min(errors)
			h_t = errors.index(min_error)				
			self.clfs_picked.append(list(self.clfs)[h_t])
			self.betas.append(0.5)	
			f = f + 0.5*np.array(list(self.clfs)[h_t].predict(features))
			pi = 1/(1+np.exp(-2*f))

	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
	