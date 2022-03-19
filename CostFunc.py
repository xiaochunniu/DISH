import numpy as np

# regularized linear regression cost function
class QuadraticCostFunc:
    def __init__(self, A, y, gamma):
        self.A = A
        self.y = y
        self.ga = gamma
        self.nclient = len(y)
        self.nsample = sum([len(y[i]) for i in range(self.nclient)])
        _, self.ndim = self.A[0].shape

    def local_func(self, x, i):
        x = x.reshape(-1, 1)
        Ai = self.A[i]
        ni, _ = Ai.shape
        yi = self.y[i].reshape((ni, 1))

        fi = 0.5 * (np.linalg.norm(Ai.dot(x) - yi)**2 / self.nsample + self.ga * np.linalg.norm(x)**2 / self.nclient)
    
        return fi

    def local_grad(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape
        x = x.reshape((self.ndim, 1))
        yi = self.y[i].reshape((ni, 1))
    
        return Ai.T.dot(Ai.dot(x) - yi) / self.nsample + self.ga * x / self.nclient

    def local_hess(self, x, i):
        Ai = self.A[i]

        return Ai.T.dot(Ai) / self.nsample + self.ga * np.identity(self.ndim) / self.nclient

    def global_func(self, x):
        f = 0
        nd = np.size(x)
        if nd == self.ndim: # global function value at the same point
            for i in range(self.nclient):
                f = f + self.local_func(x, i)
        else: # global function value when plugging local points
            for i in range(self.nclient):
                f = f + self.local_func(x[i], i)

        return f


# regularized logistic regression cost function
class LogisticCostFunc:
    def __init__(self, A, y, gamma):
        self.A = A
        self.y = y
        self.ga = gamma
        self.nclient = len(y)
        self.nsample = sum([len(y[i]) for i in range(self.nclient)])
        _, self.ndim = self.A[0].shape

    def sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1 + np.exp(-z))

    def local_func(self, x, i):
        x = x.reshape(-1, 1)
        Ai = self.A[i]
        ni, _ = Ai.shape
        yi = self.y[i].reshape((ni, 1))
        hi = self.sigmoid(Ai.dot(x))
        reg = 0.5 * self.ga * np.linalg.norm(x)**2

        fi = (- yi.T.dot(np.log(hi)) - (1 - yi).T.dot(np.log(1-hi))) / self.nsample + reg / self.nclient
    
        return fi[0][0]

    def local_grad(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape
        x = x.reshape((self.ndim, 1))
        yi = self.y[i].reshape((ni, 1))
        hi = self.sigmoid(Ai.dot(x))
        reg = self.ga * x

        gi = Ai.T.dot(hi - yi) / self.nsample + reg / self.nclient
    
        return gi

    def local_hess(self, x, i):
        Ai = self.A[i]
        ni, _ = Ai.shape
        x = x.reshape((self.ndim, 1))
        hi = self.sigmoid(Ai.dot(x))
        S = np.diag(hi.T[0]) * np.diag((1-hi).T[0])
        Hi = Ai.T.dot(S).dot(Ai) / self.nsample + self.ga * np.identity(self.ndim) / self.nclient
        #print('local hessian cond at agent ', i, ' is ', np.linalg.cond(Hi))

        return Hi

    def global_func(self, x):
        f = 0
        nd = np.size(x)
        if nd == self.ndim: # global function value at the same point
            for i in range(self.nclient):
                f = f + self.local_func(x, i)
        else: # global function value when plugging local points
            for i in range(self.nclient):
                f = f + self.local_func(x[i], i)

        return f