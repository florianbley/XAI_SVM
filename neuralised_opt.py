import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy as sp
from scipy.special import logsumexp
from matplotlib import pyplot as plt


class rbf_svm_dual(nn.Module):

    def __init__(self, X_train, y_train, alphas, gamma):
        super(rbf_svm_dual, self).__init__()

        self.init_data_params(X_train, y_train)

        self.alpha = alphas
        # support vector index where alpha is not zero
        self.support_ = np.where(self.alpha != 0)[0]
        self.SVs = self.X_train[self.support_]
        self.SV_labels = self.y_train[self.support_]

        self.SV_inds_1 = (self.alpha > 0)
        self.SV_inds_0 = (self.alpha < 0)

        self.SVs_1 = self.X_train[self.SV_inds_1]
        self.SVs_0 = self.X_train[self.SV_inds_0]
        self.gamma = gamma

        self.alpha_1 = np.abs(self.alpha[self.SV_inds_1])  # positive
        self.alpha_0 = np.abs(self.alpha[self.SV_inds_0])  # negative

        self.number_pos_SVs = int(self.SV_inds_1.sum())
        self.number_neg_SVs = int(self.SV_inds_0.sum())
        number_neurons_first_layer = int(self.number_pos_SVs * self.number_neg_SVs)

        self.hid1 = nn.Linear(self.SVs.shape[1], number_neurons_first_layer)
        self.hid2 = nn.Linear(number_neurons_first_layer, self.number_neg_SVs)
        self.hid3 = nn.Linear(self.number_neg_SVs, 1)

        self.update_params()

    def init_data_params(self, X_train, y_train):
        self.X_train = X_train
        self.X_min = X_train.min()  # careful does lead to errors for test samples
        self.X_max = X_train.max()
        self.X_min_pp = X_train.min(axis=0)  # careful does lead to errors for test samples
        self.X_max_pp = X_train.max(axis=0)
        self.y_train = y_train
        self.labels = np.unique(y_train)


    def init_svm_data(self, mod, neg_definition=False, set_gamma=None):
        self.SVs = self.X_train[mod.support_]
        self.SV_labels = self.y_train[mod.support_]
        self.b = mod.intercept_

        self.alpha = mod.dual_coef_
        if neg_definition is True:
            self.alpha = -1 * mod.dual_coef_

        self.SV_inds_1 = (self.alpha>0)[0]
        self.SV_inds_0 = (self.alpha<0)[0]

        self.SVs_1 = self.SVs[self.SV_inds_1]
        self.SVs_0 = self.SVs[self.SV_inds_0]
        self.gamma=None

        if set_gamma is None:
            self.gamma = mod.gamma
        else:
            self.gamma = set_gamma
        self.alpha_1 = np.abs(self.alpha[0, self.SV_inds_1]) # positive
        self.alpha_0 = np.abs(self.alpha[0, self.SV_inds_0]) # negative

    def init_svm_data_from_bsvm(self, svm, neg_definition=False):
        assert (self.X_train == svm.X_train).all()
        assert (self.y_train == svm.y_train).all()
        bsvm_support_ = (svm.alpha > 1e-6)[:, 0]

        self.SVs = self.X_train[bsvm_support_]
        self.SV_labels = self.y_train[bsvm_support_]
        self.b = svm.b
        self.alpha = svm.alpha[bsvm_support_][:, 0] * self.y_train[bsvm_support_]
        if neg_definition is True:
            self.alpha *= -1

        self.SV_inds_1 = (self.alpha > 0)
        self.SV_inds_0 = (self.alpha < 0)

        self.SVs_1 = self.SVs[self.SV_inds_1]
        self.SVs_0 = self.SVs[self.SV_inds_0]

        self.gamma = svm.gamma
        self.alpha_1 = np.abs(self.alpha[self.SV_inds_1])  # positive
        self.alpha_0 = np.abs(self.alpha[self.SV_inds_0])  # negative

    def prepare_weights_biases(self):

        W1 = 2 * (self.SVs_1[:, None, :] - self.SVs_0[None, :, :])   # 1-i, 0-j
        W1 = W1.reshape(-1, W1.shape[-1])
        # biases
        SV1_norms = np.linalg.norm(self.SVs_1, axis=1) ** 2
        SV0_norms = np.linalg.norm(self.SVs_0, axis=1) ** 2
        alpha1_log = np.log(self.alpha_1)
        alpha0_log = np.log(self.alpha_0)

        b0 = -SV0_norms + alpha0_log/self.gamma
        b1 = -SV1_norms + alpha1_log/self.gamma

        b = b1[:, None] - b0[None, :]
        b = b.reshape(-1)

        W2 = np.zeros((self.number_neg_SVs, int(self.number_neg_SVs * self.number_pos_SVs)))
        # for every neg SV compute sum over all pos SV
        for ind in range(W2.shape[0]):
            W2[ind, ind::self.number_neg_SVs] = 1
        return W1, W2, b

    def update_params(self):
        W1, W2, b = self.prepare_weights_biases()
        new_state_dict = self.state_dict()
        new_state_dict["hid3.bias"] *= 0
        new_state_dict["hid3.weight"] = \
            new_state_dict["hid3.weight"] * 0 + 1
        new_state_dict["hid2.bias"] *= 0
        new_state_dict["hid2.weight"] = torch.Tensor(W2)
        new_state_dict["hid1.bias"] = torch.Tensor(b)
        new_state_dict["hid1.weight"] = torch.Tensor(W1)
        self.load_state_dict(new_state_dict)

    def LSE_max(self, x):
        max_vals = []
        # Per negative SV aggregate over all positive SVs
        for row in range(self.hid2.weight.shape[0]):
            sv_inds = self.hid2.weight[row] > 0
            x_svs = x[:, sv_inds]
            max_val = torch.max(x_svs, dim=1).values
            max_vals.append(max_val)
            x[:, sv_inds] = x[:, sv_inds] - max_val[:, None]
        max_vals = torch.stack(max_vals).T
        # lse trick

        # max layer
        x = torch.exp(self.gamma*x)
        x = self.hid2(x)
        x = torch.log(x)
        x = 1 / self.gamma * x
        x = x + max_vals
        return x

    # todo checken ob loop für beide Layer funktioniert
    def LSE_min(self, x):
        min_vals = []
        # Maxima pro Neuron des zweiten Layers
        for row in range(self.hid3.weight.shape[0]):
            sv_inds = self.hid3.weight[row] > 0
            x_svs = x[:, sv_inds]
            min_val = torch.min(x_svs, dim=1).values
            min_vals.append(min_val)
            x[:, sv_inds] = x[:, sv_inds] - min_val[:, None]
        min_vals = torch.stack(min_vals).T
        # lse trick

        # max layer
        x = torch.exp(-self.gamma * x)
        x = self.hid3(x)
        x = -torch.log(x)
        x = 1 / self.gamma * x
        x = x + min_vals
        return x

    def forward(self, x_input, return_intermediates=False):
        x = x_input
        if len(x.shape) == 1:
            x = x[None, :]
        if not torch.is_tensor(x):
            x = torch.Tensor(x)

        x = self.hid1(x)
        a0_layer_out = x.clone().detach()

        x_alt = x.reshape(-1, self.number_neg_SVs, self.number_pos_SVs)
        x_alt = 1/self.gamma * torch.logsumexp(self.gamma*x_alt, dim=2)

        x = self.LSE_max(x)
        a1_layer_out = x.clone().detach()

        x = self.LSE_min(x_alt)
        a2_layer_out = x.clone().detach()

        output = self.gamma*x[:, 0]


        if return_intermediates is True:
            return output, a0_layer_out, a1_layer_out, a2_layer_out
        else:
            return output

    def forward_alt(self, x_input, return_intermediates=False):
        activations = []
        x = x_input
        if len(x.shape) == 1:
            x = x[None, :]
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        x = self.hid1(x)
        x = x.reshape(-1, self.number_neg_SVs, self.number_pos_SVs)
        pass

    def compute_gradients(self, x, classe=0):
        input = x
        input.requires_grad = True
        output = self.forward(x)[classe]
        output.backward()
        return input.grad.data

    def compute_sensitivity(self, x, classe=0):
        sensitivity = self.compute_gradients(x, classe=classe).detach().numpy() ** 2
        return sensitivity

    def get_centroid_difference(self, of_SVs=False):
        if of_SVs is False:
            ind_0 = self.y_train == self.labels[0]
            X_train_0 = self.X_train[ind_0]
            X_train_0_centroid = X_train_0.mean(axis=0)

            ind_1 = self.y_train == self.labels[1]
            X_train_1 = self.X_train[ind_1]
            X_train_1_centroid = X_train_1.mean(axis=0)

            return X_train_1_centroid - X_train_0_centroid

        elif of_SVs is True:
            SV_centroid_0 = self.alpha_0.dot(self.SVs_0)
            SV_centroid_1 = self.alpha_1.dot(self.SVs_1)

            return SV_centroid_1 - SV_centroid_0

    def compute_q_score_SVM(self, X):
        X = torch.Tensor(X)
        SVs_1 = self.SVs_1
        alpha_1 = self.alpha_1

        SVs_0 = self.SVs_0
        alpha_0 = self.alpha_0

        A_1 = (X[:, None, :] - SVs_1[None, :, :])
        A_1 = np.linalg.norm(A_1, axis=2) ** 2
        A_1 *= -self.gamma
        K_1 = np.exp(A_1)

        A_0 = (X[:, None, :] - SVs_0[None, :, :])
        A_0 = np.linalg.norm(A_0, axis=2) ** 2
        A_0 *= -self.gamma
        K_0 = np.exp(A_0)

        sum_1 = np.dot(K_1, alpha_1)
        sum_0 = np.dot(K_0, alpha_0)

        q = np.log(sum_1) - np.log(sum_0)
        return q

    def z_box_rule(self, a, W, R):
        if(not torch.is_tensor(a)):
            a = torch.Tensor(a)

        L = (W >= 0) * self.X_min_pp  # li where weights positive
        H = (W < 0) * self.X_max_pp  # hi where weights negative
        A = a[:, None, :] - L - H
        A = A * W
        A = A / (A.sum(axis=2)[:, :, None]+1e-12)
        A = A * R[:, :, None]
        R0 = A.sum(axis=1).detach().numpy()

        assert np.isclose(R0.sum(), R.detach().numpy().sum(), atol=1e-03)

        return np.abs(R0)

    def soft_max_rule(self, a, W, R):
        n = R.shape[0]
        dim2 = int(W.shape[0])  # Wieviele Neuronenrelevanzen werden propagiert?
        dim1 = int(a.shape[1] / dim2)
        a = torch.reshape(a, (n, dim1, dim2)) # eine Zeile pro Neuron im tieferen layer

        # for numerical stability subtract the maximum value, it does not change the result in propagation
        a = a - torch.max(a, dim=1).values[:, None, :]
        a_exp = torch.exp(a)

        A = (a_exp / a_exp.sum(axis=1)[:, None])
        R2 = A * R[:, None, :]
        R2 = torch.reshape(R2, (n, -1))
        return R2

    def soft_min_rule(self, a, W, R):
        n = R.shape[0]
        dim2 = int(W.shape[0])  # Wieviele Neuronenrelevanzen werden propagiert?
        dim1 = int(a.shape[1] / dim2)
        a = torch.reshape(a, (n, dim1, dim2))  # eine Zeile pro Neuron im tieferen layer

        # for numerical stability subtract the minimum value, it does not change the result in propagation
        a = a - torch.min(a, dim=1).values[:, None, :]
        a_exp_minus = torch.exp(-a)

        A = (a_exp_minus / a_exp_minus.sum(axis=1)[:, None])
        R1 = A * R[:, None, :]
        R1 = torch.reshape(R1, (n, -1))
        return R1

    def w2_rule(self, R1):
        Wl = self.hid1.weight.clone().detach().numpy()
        W = (Wl ** 2) / ((Wl ** 2).sum(axis=1)[:, None]+1e-12)
        R0 = (W.T * R1.detach().numpy()).sum(axis=1)

        # Sicherstellen, dass Propagation konservativ ist
        assert np.isclose(R0.sum(), R1.detach().numpy().sum(), atol=1e-03)

        return np.abs(R0[None, :])

    def z_rule(self, x, R1):

        W = self.hid1.weight.detach().numpy()
        b = self.hid1.bias.detach().numpy()

        a = (x.dot(W.T))[0]

        propagation_factors = (W * x)/(a[:, None]+1e-12)
        R0 = (propagation_factors * R1.detach().numpy().T).sum(0)
        return R0[None, :]


    def stabilize(self, x, epsilon=1e-6, clip=False, norm_scale=False, dim=None):
        sign = ((x == 0.) + np.sign(x))
        return x + sign * epsilon


    def gamma_rule(self, x, R1, gamma=0.1):

        weight = self.hid1.weight.detach().clone().numpy()
        bias = self.hid1.bias.detach().clone().numpy()

        '''Replicates the Gamma rule.'''
        output = x @ weight.T + bias
        #bias *= 0
        pinput = x.clip(min=0)
        ninput = x.clip(max=0)
        pwgamma = weight + weight.clip(min=0) * gamma
        nwgamma = weight + weight.clip(max=0) * gamma
        pbgamma = bias + bias.clip(min=0) * gamma
        nbgamma = bias + bias.clip(max=0) * gamma

        pgrad_out = (R1.detach().numpy()/ self.stabilize(pinput @ pwgamma.T + ninput @ nwgamma.T + pbgamma)) * (output > 0.)
        positive = pinput * (pgrad_out @ pwgamma) + ninput * (pgrad_out @ nwgamma)

        ngrad_out = (R1.detach().numpy() / self.stabilize(pinput @ nwgamma.T + ninput @ pwgamma.T + nbgamma)) * (output < 0.)
        negative = pinput * (ngrad_out @ nwgamma) + ninput * (ngrad_out @ pwgamma)

        return positive + negative


    def support_vector_rule(self, x, al, am, an, R1, rem=False):
        # Ich brauche für jedes al den Wert desjenigen am, das al als Input hatte
        # Die weights von hid2 (am_shape x al_shape) zeigen an, welche al welchen am zugeordnet sind
        # D.h. jede Spalte steht für ein al und hat exakt einen 1er Eintrag.
        # Ich multipliziere zeilenweise die Werte der ams. Die Summen pro Spalten geben somit
        # den am-Wert an der jedem al- Wert zugehörig ist
        Am = (self.hid2.weight.clone().detach().T * am).sum(axis=1)
        if not rem:
            an = np.abs(an) # todo remove
        Thetas = (an - al)

        # max trick
        Pm = self.gamma * np.exp((am-am.max(1)[:, None]) / np.sum(np.exp((am-am.max(1)[:, None]))))
        al_reshape = np.reshape(al[0], (self.number_pos_SVs, self.number_neg_SVs))
        # min trick
        Pl_sub = np.exp(-(al_reshape - al_reshape.min(0)[None, :])) / np.sum(np.exp(-(al_reshape - al_reshape.min(0)[None, :])), axis=0)[None, :] # TODO sichergehen, dass beide identisch
        #Pl_sub2 = np.exp(al_reshape-al_reshape.max(axis=0)) / np.sum(np.exp(al_reshape-al_reshape.max(axis=0)), axis=0)
        Pl = np.reshape(Pl_sub * Pm, (-1))

        assert np.isclose((al + Thetas) * Pl, np.abs(R1.detach().numpy()), atol=1e-2).all()
        # Um dieselbe Reihenfolge der i SVs sicherzustellen, nutze ich exakt dieselbe
        # Broadcastmethode wie ich sie in prepare_weights_biases genutzt habe
        Xi = (self.SVs_1[:, None, :] + np.zeros(self.SVs_0.shape)[None, :, :])
        Xi = Xi.reshape(-1, Xi.shape[-1])

        # Beta ist als plus theta durch L2 Norm von wl mal -2
        # L2 Norm ist Dot Produkt von WL mit sich selbst, die Normen pro
        # wl dimension stehen auf der Diagonalen
        Bl = self.hid1.bias.clone().detach().numpy()
        Wl = self.hid1.weight.clone().detach().numpy()
        Wl_norms = np.linalg.norm(Wl, axis=1)**2 + 1e-12
        Betas = 2 * self.gamma * ((Wl * Xi).sum(axis=-1) + Bl + Thetas[0]) / Wl_norms # vorher np.diag(Wl.dot(Xi.T))

        X_null = (Xi - ((1 / (2 * self.gamma)) * Betas[:, None] * Wl))
        R_null = Pl * ((Wl * X_null).sum(-1) + Bl + Thetas) # vorher np.diag(Wl.dot(X_null.T))
        # Sicherstellen, dass Nullpunkte null Relevanz haben
        assert np.allclose(R_null, 0, atol=1e-02)

        R_kl = (Pl[:, None] * Wl * (x - X_null))
        R_k = R_kl.sum(axis=0)

        # Sicherstellen, dass Propagation konservativ ist
        assert np.allclose(R_kl.sum(axis=1), np.abs(R1.detach().numpy()[0]), atol=1e-03)

        return R_k[None, :]

    def lrp(self, x, last_propagation="svr", rem=False, gamma=0.1):
        # TODO: Propagierung parametrisieren
        W2 = self.hid3.weight
        W1 = self.hid2.weight
        W0 = self.hid1.weight

        output, a0_layer_out, a1_layer_out, a2_layer_out = self.forward(x, return_intermediates=True)
        #assert output.detach().numpy() != 0
        R = output[:, None]

        # a2 are basically
        R2 = self.soft_max_rule(a=a1_layer_out, W=W2, R=R)
        R1 = self.soft_min_rule(a=a0_layer_out, W=W1, R=R2)

        if last_propagation == "svr":
            R0 = self.support_vector_rule(
                x=x, al=a0_layer_out.detach().numpy(), am=a1_layer_out.detach().numpy(), an=a2_layer_out.detach().numpy(), R1=R1, rem=rem)

        elif last_propagation == "zbox":
            R0 = self.z_box_rule(a=x, W=W0, R=R1)

        elif last_propagation == "w2":
            R0 = self.w2_rule(R1=R1)

        elif last_propagation == "gamma":
            R0 = self.gamma_rule(x=x, R1=R1)

        elif last_propagation == "sensitivity":
            sample = torch.Tensor(x)
            sample.requires_grad = True
            fwd = self.forward(sample)

            fwd.backward()
            grad = sample.grad
            return grad.detach().numpy() ** 2

        elif last_propagation == "z":
            R0 = self.z_rule(x=x, R1=R1)

        else:
            print("Keine richtige letzte Propagationsregel ausgewählt")
            return None

        return R0