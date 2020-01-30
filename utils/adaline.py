import numpy as np


class AdalineGD(object):
    """Adaptive Linear Neuron classifier
	---Parameters---
	eta : float - Learning rate (between 0.0 and 1.0)
	n_iter : int - Iterates over the training dataset
	---Attributes---
	w_ :  1D array - weights after fitting
	errors_ : list - number of misclassifications in every epoch
	"""

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
		X : m x n matrix where m is the number of samples and the n is the number of features
		y : vector of target values 
		"""
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        """returns class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
