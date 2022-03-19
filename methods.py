from copy import copy
from threading import local
import numpy as np
import copy
#import matplotlib.pyplot as plt

# define train_param / tune_param
class TrainParam:
    def __init__(self, alpha1 = None, beta1 = None, alpha2 = None, beta2 = None, mu = None, K = None, client_gradient = None, client_Newton = None, 
                initial_x = None):
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.mu = mu
        self.K = K
        self.client_gradient = client_gradient
        self.client_Newton = client_Newton
        self.initial_x = initial_x


# alpha1, beta1, etc are ranges of parameters to be tuned 
class TuneParam:
    def __init__(self, alpha1_range = None, beta1_range = None, alpha2_range = None, beta2_range = None, mu_range = None, K = None, client_gradient = None, client_Newton = None, 
                initial_x = None):
        self.alpha1_range = alpha1_range
        self.beta1_range = beta1_range
        self.alpha2_range = alpha2_range
        self.beta2_range = beta2_range
        self.mu_range = mu_range
        self.K = K
        self.client_gradient = client_gradient # clients doing second-order updates in DisHybrid
        self.client_Newton = client_Newton # clients doing first-order updates in DisHybrid
        self.initial_x = initial_x


# Algorithms  
# EXTRA
class EXTRA:
    def __init__(self, Z, func, fn_star, x_star):
        self.Z = Z # consensus matrix
        self.func = func
        _, self.ndim = self.func.A[0].shape
        self.nclient = len(self.func.y)
        self.fn_star = fn_star
        self.x_star = x_star

    def train(self, param):
        alpha1, K = 2**param.alpha1, param.K

        x = copy.deepcopy(param.initial_x) # deep copy to make sure the initial point is fixed
        new_x = np.zeros([self.nclient, self.ndim, 1])
        y = np.zeros([self.nclient, self.ndim, 1])
        g = np.zeros([self.nclient, self.ndim, 1])
        new_g = np.zeros([self.nclient, self.ndim, 1])

        fn = []
        x_list = []
        rel_dis_x = []

        dis_x_0 = 0
        for i in range(self.nclient):
            dis_x_0 += np.linalg.norm(x[i]-self.x_star)

        zx = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            for j in range(self.nclient):
                zx[i] += self.Z[i][j] * x[j]
            g[i] = self.func.local_grad(x[i], i)
            new_x[i] = zx[i] - alpha1 * g[i]

        for k in range(K):
            x_list.append([])
            new_zx = np.zeros([self.nclient, self.ndim, 1])
            for i in range(self.nclient):
                for j in range(self.nclient):
                    new_zx[i] += self.Z[i][j] * new_x[j]
                
                new_g[i] = self.func.local_grad(new_x[i], i)
                y[i] = new_x[i] + new_zx[i] - (x[i] + zx[i]) / 2 - alpha1 * (new_g[i] - g[i])
                #s_alpha = alpha1 / (k+1)
                #y[i] = (new_x[i] + new_zx[i]) / 2 - s_alpha * (new_g[i])
                x_list[-1].append(np.linalg.norm(x[i]-self.x_star))

            x = copy.deepcopy(new_x)
            new_x = copy.deepcopy(y)
            zx = copy.deepcopy(new_zx)
            g = copy.deepcopy(new_g)

            fn.append(self.func.global_func(x))
            rel_dis_x.append(sum(x_list[-1])/dis_x_0)
    
            if fn[-1] > 1e10:
                break
            
            #if np.log(rel_dis_x[-1]) < -13.5 and np.log(abs(fn[-1] - self.fn_star)) < -20:
            if np.log(rel_dis_x[-1]) < -17 and np.log(abs(fn[-1] - self.fn_star)) < -20:
                break

        #print('diff x         :', x[0]-self.x_star)

        return np.array(fn) - self.fn_star, k, rel_dis_x, x_list     

    def tune(self, param):
        tune_dish = []
        a_range = param.alpha1_range
        for a in a_range:
            fn_dish, k_dish, rel_dis_x, _ = self.train(param = TrainParam(alpha1 = a, K = param.K, initial_x=param.initial_x))
            # print('alpha=', a, 'fn_dish_last=', fn_dish[-1], 'k_dish=', k_dish)
            if fn_dish[-1] < 1: #and k_dish < param.K-1:
                print('alpha=', a, 'fn_dish_last=', fn_dish[-1], 'k_dish=', k_dish, 'rel_x', rel_dis_x[-1])
                tune_dish.append([a, k_dish, fn_dish, rel_dis_x])

            print('a=', a)
        
        return tune_dish


# Distributed Hybrid, when nsecond=0, it's DISH_G; when nsecond=nclient, it's DISH_N
class DISH:
    def __init__(self, Z, func, fn_star, x_star):
        self.func = func
        _, self.ndim = self.func.A[0].shape
        self.nclient = len(self.func.y)
        self.fn_star = fn_star
        self.x_star = x_star
        self.W = np.kron(np.identity(self.nclient) - Z, np.identity(self.ndim)) # W = (I_n - Z) \otimes I_d, Z is the consensus matrix

    def train(self, param):
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, param.mu, param.K

        x = param.initial_x.copy() # deep copy to make sure the initial point is fixed

        new_x = np.zeros([self.nclient, self.ndim, 1])
        dual = np.zeros([self.nclient, self.ndim, 1])
        
        fn = []
        x_list = []
        rel_dis_x = []

        dis_x_0 = 0

        for i in range(self.nclient):
            dis_x_0 += np.linalg.norm(x[i]-self.x_star)


        #zx = np.zeros([self.nclient, self.ndim, 1])
        #for i in range(self.nclient):
        #    for j in range(self.nclient):
        #       zx[i] += self.Z[i][j] * x[j]

        for k in range(K):
            x_list.append([])

            Wx = self.W.dot(x.reshape([self.nclient * self.ndim, 1]))
            Wx = Wx.reshape([self.nclient, self.ndim, 1])
            Wt_dual = np.transpose(self.W).dot(dual.reshape([self.nclient * self.ndim, 1]))
            Wt_dual = Wt_dual.reshape([self.nclient, self.ndim, 1])

            for i in param.client_gradient: 
                gi = self.func.local_grad(x[i], i) # local gradient
    
                new_x[i] = x[i] - alpha1 * (gi + Wt_dual[i] + mu * Wx[i])
            
                dual[i] = dual[i] + beta1 * Wx[i]
    
                x[i] = copy.deepcopy(new_x[i])

                x_list[k].append(np.linalg.norm(x[i]-self.x_star))


            for i in param.client_Newton: 
                gi = self.func.local_grad(x[i], i) # local gradient
                Hi = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
    
                new_x[i] = x[i] - alpha2 * np.asmatrix(np.linalg.inv(Hi)) * (gi + Wt_dual[i] + mu * Wx[i])
                
                dual[i] = dual[i] + beta2 * np.asmatrix(Hi) * Wx[i]

                x[i] = copy.deepcopy(new_x[i])

                x_list[-1].append(np.linalg.norm(x[i]-self.x_star))              
                
            fn.append(self.func.global_func(x))
            rel_dis_x.append(sum(x_list[-1])/dis_x_0)
    
            if fn[-1] > 1e10:
                break
            
            #if np.log(rel_dis_x[-1]) < -13.5 and np.log(abs(fn[-1] - self.fn_star)) < -20:
            if np.log(rel_dis_x[-1]) < -17 and np.log(abs(fn[-1] - self.fn_star)) < -20:
                break
    
        return np.array(fn) - self.fn_star, k, rel_dis_x, x_list      

    def tune(self, param):
        tune_dish = []
        a_range, b_range, a2_range, b2_range, mu_range = param.alpha1_range, param.beta1_range, param.alpha2_range, param.beta2_range, param.mu_range
        for a in a_range:
            for b in b_range:
                for a2 in a2_range:
                    for b2 in b2_range:
                        for u in mu_range:
                            fn_dish, k_dish, rel_dis_x, _ = self.train(
                                param = TrainParam(
                                    alpha1 = a, beta1 = b, alpha2 = a2, beta2 = b2, mu = u, K = param.K, 
                                    client_gradient= param.client_gradient, client_Newton = param.client_Newton, initial_x=param.initial_x
                                    )
                                )
                            if np.log(rel_dis_x[-1]) < -5: #and k_dish < param.K-1:
                                print('alpha=', a, 'beta=', b, 'alpha2=', a2, 'beta2=', b2, 'mu=', u, 'fn_dish_last=', fn_dish[-1], 'k_dish=', k_dish, 'rel_x', rel_dis_x[-1])
                                tune_dish.append([a, b, a2, b2, u, k_dish, fn_dish])
                                #plt.plot(np.log(fn_dish))
                                #plt.savefig('Logistic_Synthetic/tune/DisH' + str(len(param.client_Newton)) + '/tune' + str(a) + '_' + str(b) + '_' + str(a2) + '_' + str(b2) + '_' + str(u) + '.pdf')
                                #plt.clf()
            print('a1=', a)
        
        return tune_dish

 
# ESOM (Second-order primal and first-order dual)
class ESOM:
    def __init__(self, Z, func, fn_star, x_star, ESOM_K):
        self.Z = Z # consensus matrix
        self.func = func
        _, self.ndim = self.func.A[0].shape
        self.nclient = len(self.func.y)
        self.fn_star = fn_star
        self.x_star = x_star
        self.ESOM_K = ESOM_K

    def train(self, param):
        alpha, mu, T = 2**param.alpha2, param.mu, param.K

        x = param.initial_x.copy() # deep copy to make sure the initial point is fixed
        dual = np.zeros([self.nclient, self.ndim, 1])
        
        fn = []
        x_list = []
        rel_dis_x = []

        dis_x_0 = 0
        for i in range(self.nclient):
            dis_x_0 += np.linalg.norm(x[i]-self.x_star)
        
        # B blocks
        B = np.zeros([self.nclient, self.nclient])
        for i in range(self.nclient):
            for j in range(self.nclient):
                if i == j:
                    B[i][i] = alpha * (1 - self.Z[i][i])
                else:
                    B[i][j] = alpha * self.Z[i][j]

        zx = np.zeros([self.nclient, self.ndim, 1])
        for i in range(self.nclient):
            for j in range(self.nclient):
                zx[i] += self.Z[i][j] * x[j]
        
        for t in range(T): 
            x_list.append([])

            d = np.zeros([self.nclient, self.ndim, 1])
            
            for i in range(self.nclient):
                # calculate the D block and gradients
                Di = self.func.local_hess(x[i], i) + (2 * alpha * (1 - self.Z[i][i]) + mu) * np.identity(self.ndim)          
                gi = self.func.local_grad(x[i], i) + dual[i] + alpha * (x[i] - zx[i])
                d[i] = - np.asmatrix(np.linalg.inv(Di)) * gi
            
                for k in range(self.ESOM_K):
                    Bd = np.zeros([self.nclient, self.ndim, 1])
                    for i in range(self.nclient):
                        for j in range(self.nclient):
                            Bd[i] += B[i][j] * d[j]
                    
                    for i in range(self.nclient):
                        d[i] = np.asmatrix(np.linalg.inv(Di)) * (Bd[i] - gi)

                x[i] = x[i] + d[i]

                x_list[-1].append(np.linalg.norm(x[i]-self.x_star))
            
            # communicate with neighbors and dual updates
            zx = np.zeros([self.nclient, self.ndim, 1])
            for i in range(self.nclient):
                for j in range(self.nclient):
                    zx[i] += self.Z[i][j] * x[j]
            
            for i in range(self.nclient):
                dual[i] = dual[i] + alpha * (x[i] - zx[i])       
        
            fn.append(self.func.global_func(x))
            rel_dis_x.append(sum(x_list[-1])/dis_x_0)

            if fn[-1] > 1e10:
                break
            
            #if np.log(rel_dis_x[-1]) < -13.5 and np.log(abs(fn[-1] - self.fn_star)) < -20:
            if np.log(rel_dis_x[-1]) < -17 and np.log(abs(fn[-1] - self.fn_star)) < -20:
                break

        return np.array(fn) - self.fn_star, t, rel_dis_x, x_list 

    def tune(self, param):
        tune_esom = []
        a_range, mu_range = param.alpha2_range, param.mu_range
        for a in a_range:
            for u in mu_range:
                fn_esom, k_esom, rel_dis_x, _ = self.train(param = TrainParam(alpha2 = a, mu = u, K = param.K, initial_x = param.initial_x))
                if np.log(rel_dis_x[-1]) < -10:
                    print('alpha=', a, 'mu=', u, 'fn_esom_last=', fn_esom[-1], 'k_esom=', k_esom, 'rel_dis_x', rel_dis_x[-1])
                    tune_esom.append([a, u, k_esom, fn_esom])
                    #plt.plot(np.log(fn_esom))
                    #plt.savefig('Quadratic_Synthetic/LargeLocalCond/tune/esom' + '/tune' + str(a) + '_' + str(b) + '_' + str(u) + '.pdf')
                    #plt.clf()
        
        return tune_esom


# Distributed Hybrid, when nsecond=0, it's DISH_G; when nsecond=nclient, it's DISH_N
class DISH_G_and_N:
    def __init__(self, Z, func, fn_star, x_star, local_t, local_init):
        self.func = func
        _, self.ndim = self.func.A[0].shape
        self.nclient = len(self.func.y)
        self.fn_star = fn_star
        self.x_star = x_star
        self.W = np.kron(np.identity(self.nclient) - Z, np.identity(self.ndim)) # W = (I_n - Z) \otimes I_d, Z is the consensus matrix
        self.local_t = local_t
        self.local_init = local_init

    def train(self, param):
        alpha1, beta1, alpha2, beta2, mu, K = 2**param.alpha1, 2**param.beta1, 2**param.alpha2, 2**param.beta2, param.mu, param.K

        x = param.initial_x.copy() # deep copy to make sure the initial point is fixed

        new_x = np.zeros([self.nclient, self.ndim, 1])
        dual = np.zeros([self.nclient, self.ndim, 1])
        
        fn = []
        x_list = []
        rel_dis_x = [1]

        dis_x_0 = 0
        iter_k = 0

        for i in range(self.nclient):
            dis_x_0 += np.linalg.norm(x[i]-self.x_star) 

        # local update type for each agent at each iteration
        local_type = np.zeros([self.nclient, K])
        for i in range(self.nclient):
            if self.local_init[i] == 1:
                first = [1]
                switch = [2]
            else:
                first = [2]
                switch = [1]

            ti = self.local_t[i]
            m = K // (2 * ti)
            r = K - m * 2 * ti
            arr = (first * ti + switch * ti) * m
            if r <= ti:
                arr += first * r
            else:
                arr += first * ti + switch * (r-ti)
            local_type[i] = np.array(arr)
        for k in range(iter_k+1, K):
            x_list.append([])

            Wx = self.W.dot(x.reshape([self.nclient * self.ndim, 1]))
            Wx = Wx.reshape([self.nclient, self.ndim, 1])
            Wt_dual = np.transpose(self.W).dot(dual.reshape([self.nclient * self.ndim, 1]))
            Wt_dual = Wt_dual.reshape([self.nclient, self.ndim, 1])

            for i in range(self.nclient): 
                gi = self.func.local_grad(x[i], i) # local gradient
                if local_type[i][k] == 1:
                    new_x[i] = x[i] - alpha1 * (gi + Wt_dual[i] + mu * Wx[i])
                    dual[i] = dual[i] + beta1 * Wx[i]

                if local_type[i][k] == 2:
                    Hi = self.func.local_hess(x[i], i) + mu * np.identity(self.ndim) # local Hessian
                    new_x[i] = x[i] - alpha2 * np.asmatrix(np.linalg.inv(Hi)) * (gi + Wt_dual[i] + mu * Wx[i])             
                    dual[i] = dual[i] + beta2 * np.asmatrix(Hi) * Wx[i]

                x[i] = copy.deepcopy(new_x[i])

                x_list[-1].append(np.linalg.norm(x[i]-self.x_star))              
                
            fn.append(self.func.global_func(x))
            rel_dis_x.append(sum(x_list[-1])/dis_x_0)
    
            if fn[-1] > 1e10:
                break
            
            #if np.log(rel_dis_x[-1]) < -13.5:# and np.log(abs(fn[-1] - self.fn_star)) < -20:
            if np.log(rel_dis_x[-1]) < -17:
                break
    
        return np.array(fn) - self.fn_star, k, rel_dis_x, x_list      

    def tune(self, param):
        tune_dish = []
        a_range, b_range, a2_range, b2_range, mu_range = param.alpha1_range, param.beta1_range, param.alpha2_range, param.beta2_range, param.mu_range
        for a in a_range:
            for b in b_range:
                for a2 in a2_range:
                    for b2 in b2_range:
                        for u in mu_range:
                            fn_dish, k_dish, rel_dis_x, _ = self.train(
                                param = TrainParam(
                                    alpha1 = a, beta1 = b, alpha2 = a2, beta2 = b2, mu = u, K = param.K, 
                                    client_gradient= param.client_gradient, client_Newton = param.client_Newton, initial_x=param.initial_x
                                    )
                                )
                            if np.log(rel_dis_x[-1]) < -13.5 and k_dish < param.K-1:
                                print('alpha=', a, 'beta=', b, 'alpha2=', a2, 'beta2=', b2, 'mu=', u, 'fn_dish_last=', fn_dish[-1], 'k_dish=', k_dish, 'rel_x', rel_dis_x[-1])
                                #tune_dish.append([a, b, a2, b2, u, k_dish, fn_dish])
                                #plt.plot(np.log(fn_dish))
                                #plt.savefig('Logistic_Synthetic/tune/DisH' + str(len(param.client_Newton)) + '/tune' + str(a) + '_' + str(b) + '_' + str(a2) + '_' + str(b2) + '_' + str(u) + '.pdf')
                                #plt.clf()
            print('a1=', a)
        
        return tune_dish