from curses import A_REVERSE
import numpy as np
import random

from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import json
import pandas as pd

# parameters to determine the distribution of sampling
class SampleParam: 
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

# generate synthetic data, linear regression model
class QuadraticDataGen: 
    def __init__(self, nclient, ndim, nsample_param, scaling_param, prior_param):
        self.nclient = nclient
        self.ndim = ndim
        self.nsample_param = nsample_param
        self.scaling_param = scaling_param
        self.prior_param = prior_param

    def run(self):
        print('local datasizes')
        #ni = np.random.lognormal(self.nsample_param.mean, self.nsample_param.sigma, self.nclient).astype(int) + 5 # number of samples per client
        ni = np.ones(self.nclient) * 50
        ni = [int(i) for i in ni]
        print(ni)

        A = [[] for _ in range(self.nclient)]
        y = [[] for _ in range(self.nclient)]

        #### define some eprior ####
        x_0 = np.random.normal(self.prior_param.mean, self.prior_param.sigma, self.ndim)
        #scaling = np.random.lognormal(self.scaling_param.mean, self.scaling_param.sigma, self.nclient) # how diverse the data is
        #scaling = scaling/sum(scaling)
        
        for i in range(self.nclient):
            #ai = np.random.rand(self.ndim, 1)
            ai = np.ones([self.ndim, 1])
            ai[:int(self.ndim/2)] = [ai[i]*10 for i in range(int(self.ndim/2))]
            ai[int(self.ndim/2):] = [ai[i+int(self.ndim/2)]*1e-1 for i in range(self.ndim-int(self.ndim/2))]
            Ai = np.random.normal(size = [ni[i], self.ndim]).dot(np.diag(ai.T[0])) #* scaling[i] 
            print(np.linalg.cond(Ai))
            
            sig = np.random.random(1)
            v = np.random.normal(0, sig, ni[i]) 
            yi = Ai.dot(x_0) + v
            yi = yi.reshape([ni[i], 1])
            A[i] = Ai
            y[i] = yi

        return A, y   


# generate synthetic data, logistic regression model
class LogisticDataGen:
    def __init__(self, nclient, ndim, nclass, nsample_param, scaling_param, prior_param):
        self.nclient = nclient
        self.ndim = ndim
        self.nclass = nclass
        self.nsample_param = nsample_param
        self.scaling_param = scaling_param
        self.prior_param = prior_param

    def softmax(self, z):
        ex = np.exp(z)
        sum_ex = np.sum(np.exp(z))
        return ex/sum_ex

    def run(self):
        ni = 50 #np.random.lognormal(self.nsample_param.mean, self.nsample_param.sigma, self.nclient).astype(int) + 50 # number of samples per client

        A = [[] for _ in range(self.nclient)]
        y = [[] for _ in range(self.nclient)]

        #### define some prior ####
        mean_W = np.random.normal(self.prior_param.mean, self.prior_param.sigma, self.nclient)
        mean_b = mean_W
        mean_x = np.random.normal(self.prior_param.mean, self.prior_param.sigma, self.nclient)

        mean_x = np.zeros((self.nclient, self.ndim))
        for i in range(self.nclient):
            mean_x[i] = np.random.normal(mean_x[i], 1, self.ndim)

        #diagonal = np.zeros(self.ndim)
        #for j in range(self.ndim):
        #    diagonal[j] = np.power((j+1), -1.2)
        #cov_x = np.diag(diagonal)

        for i in range(self.nclient):

            W = np.random.normal(mean_W[i], 1, (self.ndim, self.nclass))
            b = np.random.normal(mean_b[i], 1,  self.nclass)

            ai = np.ones([self.ndim, 1])
            ai[:int(self.ndim/2)] = [ai[i]*10 for i in range(int(self.ndim/2))]
            ai[int(self.ndim/2):] = [ai[i+int(self.ndim/2)]*1e-1 for i in range(self.ndim-int(self.ndim/2))]

            Ai = np.random.multivariate_normal(mean_x[i], np.identity(self.ndim), ni).dot(np.diag(ai.T[0]))
            yi = np.zeros(ni)

            for j in range(ni):
                tmp = np.dot(Ai[j], W) + b
                yi[j] = np.argmax(self.softmax(tmp))

            A[i] = Ai#.tolist()
            y[i] = yi#.tolist()

        return A, y