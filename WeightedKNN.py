from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
import numpy as np

class WeightedKNN(BaseEstimator, ClassifierMixin):
    """Weighted KNN Classifier for Imbalanced Classes. Full API req'd for GridSearch not implemented"""

    def __init__(self, n_neighbors=5, weight=1, minority_class=1, algorithm='auto'):
        self.n_neighbors = n_neighbors
        self.weight = weight
        self.minority_class = minority_class
        self.algorithm = algorithm

    def fit(self, X, y):
        assert (self.n_neighbors % 2 != 0), "just make n_neighbors odd"
        assert (type(self.weight) == int or type(self.weight) == float), "intValue parameter must be a number"
        assert (self.minority_class in y), "minority_class parameter must a valid class"

        self.nbrs_ = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm=self.algorithm).fit(X)
        self.labels_ = y

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "nbrs_")
            getattr(self, "labels_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        distances_, indices = self.nbrs_.kneighbors(X)
        y_labels = self.labels_[indices]

        pred = [self.weighted_vote(nbr_classes, self.weight) for nbr_classes in y_labels]
        return np.array(pred, dtype=int)

    def weighted_vote(self, nbr_classes, weight):  # weight is given to minority class
        """Just gonna assume class labels 0 and 1 here. No need to generalize for now..."""
        return 1 if np.sum(nbr_classes) * weight > len(nbr_classes) / 2.0 else 0
