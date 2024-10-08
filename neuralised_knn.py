from neuralised_svm import neuralised_svm
import numpy as np
import sklearn

class knn_svm_abstraction:
    def __init__(self, x_sup_pos, x_sup_neg, gamma=1e-5):
        self.support_vectors_ = np.concatenate([x_sup_pos, x_sup_neg])
        self.gamma = gamma
        self.intercept_ = 0
        self.dual_coef_ = np.concatenate([np.ones([len(x_sup_pos)]), -np.ones([len(x_sup_neg)])])[None]

class neuralised_knn(neuralised_svm):
    def __init__(self, knn):
        self.X_train = knn._fit_X
        self.y_train = knn._y
        self.k = knn.n_neighbors
        self.num_pos = self.k # TODO rename
        self.num_neg = len(self.X_train) - self.k # TODO rename
        self.gamma = 1e-5
        self.intercept_ = 0

        unique_targets = np.unique(self.y_train)
        # take the larger class label as the positive class
        self.pos_class = np.max(unique_targets)
        self.neg_class = np.min(unique_targets)

        self.x_sup_pos = self.X_train[self.y_train.ravel() == self.pos_class]
        self.x_sup_neg = self.X_train[self.y_train.ravel() == self.neg_class]


    def update_neuralisation_parameters(self, x):
        D_explained_train = sklearn.metrics.pairwise.euclidean_distances(x, self.X_train) ** 2
        D_train_pos = D_explained_train[0, self.y_train.ravel() == self.pos_class]
        D_train_neg = D_explained_train[0, self.y_train.ravel() == self.neg_class]

        cutoff_ind = int((self.n_best_neighbors + 1) / 2)
        pos_neighbor_inds = np.argsort(D_train_pos)[:cutoff_ind]
        neg_neighbor_inds = np.argsort(D_train_neg)[:cutoff_ind]

        x_sup_pos = self.X_train[self.y_train.ravel() == self.pos_class][pos_neighbor_inds]
        x_sup_neg = self.X_train[self.y_train.ravel() == self.neg_class][neg_neighbor_inds]

        self.support_vectors_ = np.concatenate([x_sup_pos, x_sup_neg])
        self.dual_coef_ = np.concatenate([np.ones([len(x_sup_pos)]), -np.ones([len(x_sup_neg)])])[None]

        self.alphas_pos = self.dual_coef_[0][self.dual_coef_[0] > 0]
        self.alphas_neg = np.abs(self.dual_coef_[0][self.dual_coef_[0] < 0])

        self.x_sup_pos = self.support_vectors_[self.dual_coef_[0] > 0]
        self.x_sup_neg = self.support_vectors_[self.dual_coef_[0] < 0]


    def compute_point_pair_weights(self, x, with_intercept: bool=False, kappa: int = 0):
        """
        In neuralised KNN, the "support vectors" of both classes are the (k+1)/2 nearest neighbors of the query point
        for each class. In contrast to neuralised SVMs, the point pair weights are not computed with a softmax. Instead,
        they are simply one over the number of support vectors of the respective class.
        """
        n_points_per_class = (self.k + 1) // 2 + kappa
        weight = 1 / n_points_per_class

        # now get an index vector for each class marking the n_points_per_class nearest neighbors
        # of the query points

        # get the distances of the query point to the training data
        D_explained_train = sklearn.metrics.pairwise.euclidean_distances(x, self.X_train) ** 2
        D_train_pos = D_explained_train[:, self.y_train.ravel() == self.pos_class]
        D_train_neg = D_explained_train[:, self.y_train.ravel() == self.neg_class]

        # get the indices of the n_points_per_class nearest neighbors
        p_pos = np.zeros((len(x), len(self.x_sup_pos)))
        p_neg = np.zeros((len(x), len(self.x_sup_neg)))
        # set to weight if the point is a nearest neighbor
        index_closest_neighbors_pos = np.argsort(D_train_pos, axis=1)[:, :n_points_per_class]
        index_closest_neighbors_neg = np.argsort(D_train_neg, axis=1)[:, :n_points_per_class]

        # set the weights for the nearest neighbors to the weight value
        np.put_along_axis(p_pos, index_closest_neighbors_pos, weight, axis=1)
        np.put_along_axis(p_neg, index_closest_neighbors_neg, weight, axis=1)

        return p_pos, p_neg


