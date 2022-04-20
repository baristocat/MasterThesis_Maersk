import warnings
import numpy as np , pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
warnings.filterwarnings('ignore')

#simulating static NBNB data
#parameterization is done through MLE

'''
https://anton-granik.medium.com/fitting-and-visualizing-a-negative-binomial-distribution-in-python-3cc27fbc7ecf
'''
def p2mu(p_series , n):
    mu_series = [p*n/(1-p)+1 for p in p_series]
    return mu_series

def mu2p(mu_series,n):
    #p_series = [mu-1/(mu+n-1) for mu in mu_series]
    #p_series = [mu ** 2 /n + mu for mu in mu_series]
    p_series = [mu / ((mu **2 / n) + mu) for mu in mu_series]
    return p_series

def AR_process(nobs, theta, seed, var_eps , constant):
    np.random.seed(seed)
    # x is an arbitrary random variable which follows an AR-process
    x_series = np.zeros(nobs)
    epsilon = np.random.normal(0, var_eps ** 0.5, [nobs, 1]).flatten()

    for t in range(1, nobs):
        x_series[t] = constant + theta * x_series[t - 1] + epsilon[t]

    #logistic transformation into probabilities
    p_series = [1/(1+np.exp(-x)) for x in x_series]
    return p_series

def generate_NBNB_data(p_series_M,p_series_Q,n_Q, n_M,seed):
    np.random.seed(seed)

    interarr_series = []
    for i in range(len(p_series_Q)):
        interarr_series.append(np.random.negative_binomial(n_Q, p_series_Q[i], 1))

    demand_size_series = []
    for i in range(len(p_series_M)):
        new_demand = np.random.negative_binomial(n_M, p_series_M[i], 1)
        if new_demand > 0:
            demand_size_series.append(new_demand)
        else:
            demand_size_series.append(1)

    y = []
    for i in range(len(p_series_Q)):
        for j in range(interarr_series[i][0]):
            y.append(0)
        y.append(demand_size_series[i])
    return y

def generate_synth_series(N,L,seed):
    #experiment parameters
    n_seeds = N

    #time series parameters
    #nobs = 2000
    n_issues = L
    n_interarr = L

    #AR-parameters
    var_eps = 0.5
    AR_theta = 0.5
    constant_M = 0
    constant_Q = 0

    #Stochastic model parameters
    n_Q = 3
    n_M = 3
    #p = 0.25

    pest_list = []
    nest_list = []
    #generate the varying p-vals
    p_series_M = AR_process(n_issues, AR_theta, seed, var_eps , constant_M)
    p_series_Q = AR_process(n_interarr, AR_theta, seed, var_eps, constant_Q)

    #generate the time series
    y = generate_NBNB_data(p_series_M,p_series_Q,n_Q, n_M,seed)

    #split the time series into Q (interarrival times) and M (demand sizes)
    M_list = [x for x in y if x > 0]
    Q_list = []

    inter_count = 0
    tot_count = 0
    for i in range(len(y)):
        if y[i] == 0:
            inter_count +=1
        #elif inter_count>0:
        else:
            tot_count += 1
            Q_list.append(inter_count)
            inter_count = 0
    return y , Q_list,M_list,p_series_Q,p_series_M