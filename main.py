import numpy as np
import random
import pandas as pd
import argparse
from scipy.optimize import minimize

from Graph import GraphGen, CompleteGraph
from Dataset import SampleParam, QuadraticDataGen, LogisticDataGen
from CostFunc import QuadraticCostFunc, LogisticCostFunc
from methods import TrainParam, TuneParam, EXTRA, DISH, ESOM, DISH_G_and_N

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.graph == 'Complete':
        data_generator = CompleteGraph(args.nclient)
    
    if args.graph == 'ER':
        data_generator = GraphGen(args.nclient, args.prob_edge)
    
    Z = data_generator.run()
    print('Consensus Z :', np.linalg.eigvals(Z))

    if args.dataset =='Quadratic_Synthetic':
        data_generator = QuadraticDataGen(nclient=args.nclient, 
                             ndim=args.ndim, 
                             nsample_param=SampleParam(args.mean_nsample_param, args.sigma_nsample_param), 
                             scaling_param=SampleParam(args.mean_scaling_param, args.sigma_scaling_param), 
                             prior_param=SampleParam(args.mean_prior_param, args.sigma_prior_param),
                             )

    if args.dataset =='Logistic_Synthetic':
        data_generator = LogisticDataGen(nclient=args.nclient, 
                             ndim=args.ndim,
                             nclass=args.nclass,
                             nsample_param=SampleParam(args.mean_nsample_param, args.sigma_nsample_param), 
                             scaling_param=SampleParam(args.mean_scaling_param, args.sigma_scaling_param), 
                             prior_param=SampleParam(args.mean_prior_param, args.sigma_prior_param),
                             )   
    
    A, y = data_generator.run()

    if args.function == 'Quadratic':
        func = QuadraticCostFunc(A, y, gamma=args.gamma)

    if args.function == 'Logistic':
        func = LogisticCostFunc(A, y, gamma=args.gamma)


    # initial point
    #initial_x = np.random.rand(args.nclient, args.ndim, 1)
    initial_x = np.zeros([args.nclient, args.ndim, 1])
    client_gradient = None
    client_Newton = None

    # calculate the optimal fn_star
    fn_min = minimize(func.global_func, np.random.rand(args.ndim, 1), tol=1e-30)
    fn_star = fn_min.fun
    x_star = np.reshape(fn_min.x, [args.ndim, 1])
    print('x_star', x_star)
    x = np.zeros([args.nclient, args.ndim, 1])
    for i in range(args.nclient):
        x[i] = x_star.reshape([args.ndim, 1])

    if args.method == 'EXTRA':
        method = EXTRA(Z, func, fn_star, x_star)

    if args.method == 'DISH':
        client_Newton = random.sample(range(args.nclient), args.nsecond) # clients that perform 2nd updates
        client_gradient = np.setdiff1d(range(args.nclient), client_Newton) # clients that perform 1st updates
        print('client_gradient', client_gradient, 'client_Newton', client_Newton)

        method = DISH(Z, func, fn_star, x_star)

    if args.method == 'ESOM':
        method = ESOM(Z, func, fn_star, x_star, args.ESOM_K)

    if args.method == 'DISH_G_and_N':
        #local_t = [random.randint(2, 100) for i in range(args.nclient)]
        local_t = np.random.lognormal(4, 2, args.nclient).astype(int) + 30
        print('local_t', local_t)
        random.seed(20222022)
        local_init = [random.randint(1, 2) for i in range(args.nclient)]
        print('local_init', local_init)
        method = DISH_G_and_N(Z, func, fn_star, x_star, local_t, local_init)

    if args.mode == "train":
        fn_list, k_iter, rel_dis_x, x_list = method.train(param = TrainParam(alpha1=args.alpha1,
                                                        beta1=args.beta1,
                                                        alpha2=args.alpha2,
                                                        beta2=args.beta2,
                                                        mu=args.mu,
                                                        K=args.K,
                                                        client_gradient=client_gradient,
                                                        client_Newton=client_Newton,
                                                        initial_x=initial_x
                                                        ))
        
        # save data
        #df = pd.DataFrame(fn_list)
        #df.to_csv(args.dataset + '/' + args.graph + '/Result/fn/' + args.method + '_' + str(args.alpha1) + '_' + str(args.beta1) + '_' 
        #            + str(args.alpha2) + '_' + str(args.beta2) + '_' + str(args.mu) + '_' + str(args.nsecond) 
        #            + '_' + str(args.seed) + '_' + str(client_Newton) + '.csv')
        df_x = pd.DataFrame(rel_dis_x)
        df_x.to_csv(args.dataset + '/' + args.graph + '/Result/' + args.method + '_' + str(args.alpha1) + '_' + str(args.beta1) + '_' 
                    + str(args.alpha2) + '_' + str(args.beta2) + '_' + str(args.mu) + '_' + str(args.nsecond) 
                    + '_' + str(args.seed) + '_' + str(client_Newton) + '.csv')
        #df_xi = pd.DataFrame(x_list)
        #df_xi.to_csv(args.dataset + '/Result/local_x/' + args.method + '_' + str(args.alpha1) + '_' + str(args.beta1) + '_' 
        #            + str(args.alpha2) + '_' + str(args.beta2) + '_' + str(args.mu) + '_' + str(args.nsecond) 
        #            + '_' + str(args.seed) + '_' + str(client_Newton) + '.csv')
        
        print('k_iter', k_iter, 'fn_last', fn_list[-1], 'rel_x', rel_dis_x[-1])

    if args.mode == "tune":
        tune_result = method.tune(param = TuneParam(alpha1_range=args.alpha1_range,
                                                    beta1_range=args.beta1_range,
                                                    alpha2_range=args.alpha2_range,
                                                    beta2_range=args.beta2_range,
                                                    mu_range=args.mu_range,
                                                    K=args.K,
                                                    client_gradient=client_gradient,
                                                    client_Newton=client_Newton,
                                                    initial_x=initial_x
                                                    ))
        # save data
        df = pd.DataFrame(tune_result)
        df.to_csv(args.dataset + '/tune/' + args.method + '_' + str(args.nsecond) + '_' + str(args.seed) + '_tune.csv')


if __name__ == '__main__':
    # parser start
    parser = argparse.ArgumentParser(description='PyTorch')

    parser.add_argument('--dataset', type=str, default='Quadratic_Synthetic') # 'Quadratic_Synthetic', 'Logistic_Synthetic'
    parser.add_argument('--function', type=str, default='Quadratic') # 'Quadratic', 'Logistic'
    parser.add_argument('--graph', type=str, default='ER') # 'Complete', 'ER'

    parser.add_argument('--nclient', type=int, default=10) # 10 for 'Quadratic_Synthetic' and 20 for 'Logistic_Synthetic'
    parser.add_argument('--prob_edge', type=int, default=0.7) # 0.7 for 'Quadratic_Synthetic' and 0.5 for 'Logistic_Synthetic'
    parser.add_argument('--ndim', type=int, default=5) # 5 for 'Quadratic_Synthetic', 3 for 'Logistic_Synthetic'
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--mean_nsample_param', type=float, default=4)
    parser.add_argument('--sigma_nsample_param', type=float, default=2)
    parser.add_argument('--mean_scaling_param', type=float, default=2) # 2 for quadratic 
    parser.add_argument('--sigma_scaling_param', type=float, default=4) # 4 for quadratic
    parser.add_argument('--mean_prior_param', type=float, default=0)
    parser.add_argument('--sigma_prior_param', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1) # penalty

    parser.add_argument('--method', type=str, default='DISH') # 'EXTRA', 'DISH', 'ESOM'. 'DISH_G_and_N'
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--alpha1', type=float, default=None)
    parser.add_argument('--beta1', type=float, default=None)
    parser.add_argument('--alpha2', type=float, default=None)
    parser.add_argument('--beta2', type=float, default=None)
    parser.add_argument('--mu', type=float, default=None)
    parser.add_argument('--K', type=int, default=500)
    parser.add_argument('--nsecond', type=int, default=None)
    parser.add_argument('--ESOM_K', type=int, default=0)

    # for tuning
    parser.add_argument('--alpha1_range', nargs='+', type=float, default=None)
    parser.add_argument('--beta1_range', nargs='+', type=float, default=None)
    parser.add_argument('--alpha2_range', nargs='+', type=float, default=None)
    parser.add_argument('--beta2_range', nargs='+', type=float, default=None)
    parser.add_argument('--mu_range', nargs='+', type=float, default=None)

    args = parser.parse_args()
    # parser end

    main(args)