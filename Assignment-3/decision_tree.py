import numpy as np
from typing import List
from classifier import Classifier

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			w = np.zeros(branches.shape[1])
			for i in range(branches.shape[1]):
				branch = branches[:,i]/np.sum(branches[:,i])
				# print('branching',branch)
				branch[np.where(branch == 0)[0]] += 1
				# print('branching + 1',branch)
				log = np.log2(branch)
				# print('log',log)
				# log[log == -np.inf] = 0
				for j in range(len(branch)):
					if log[j] != 0:
						w[i] -= branch[j]*log[j]
			# 			print('wi',w[i])
			# print('w',w)
			# print('multiply by ', np.sum(branches,axis=0)/np.sum(branches))
			cond_entropy = np.dot(np.transpose(w),np.sum(branches,axis=0)/np.sum(branches))
			# print('cond_entropy',cond_entropy)
			return cond_entropy
		
		



		features = np.array(self.features)
		# print('features',features)
		entropies = []
		for idx_dim in range(len(features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the 
		############################################################
			feature = features[:,idx_dim]
			branches = []
			for m in np.unique(feature):
				m_features = feature[np.where(feature==m)]
				m_labels = np.array(self.labels)[np.where(feature==m)]
				branch = []
				for i in range(self.num_cls):
					branch.append(np.sum(m_labels==i))
				branches.append(branch)

			# print('branches',np.array(branches).T)
			# print('entropy',conditional_entropy(np.array(branches).T.tolist()))
			entropies.append(conditional_entropy(np.array(branches).T.tolist()))
		# print('entropies',entropies)
		self.dim_split = np.argmin(entropies)
		# print('column to split on: ',self.dim_split)
		feature = features[:,self.dim_split]
		self.feature_uniq_split = np.unique(feature).tolist()
		# print('self.featureuniq',self.feature_uniq_split)
		if len(np.unique(feature)) > 1:
			for m in np.unique(feature):
				self.children.append(TreeNode(features[np.where(feature==m)].tolist(),np.array(self.labels)[np.where(feature==m)].tolist(), self.num_cls))
		else:
			self.splittable = False
		############################################################
		# TODO: split the node, add child nodes
		############################################################
		# print('am i splittable? ',self.splittable)

		# split the child nodes
		for child in self.children:
			# print('child with',child.features, ' splittable? ',child.splittable)
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		# print('self splittable?',self.splittable)
		# print('then ',self.features)
		if self.splittable:
			# print(feature)
			# print('feature: ',feature)
			# print('selffeatureuniq',self.feature_uniq_split)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			# print('idx',idx_child)
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max