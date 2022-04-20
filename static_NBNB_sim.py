import warnings
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import statsmodels.api as sm
warnings.filterwarnings('ignore')

#simulating static NBNB data
#parameterization is done through MLE

'''
simulating static NBNB data for interarrival times and demand sizes iid
dynamics are introduced by changing p-values by an AR(1)-process - the n-parameters of the NB dist is constant

https://anton-granik.medium.com/fitting-and-visualizing-a-negative-binomial-distribution-in-python-3cc27fbc7ecf
'''

def generate_NBNB_data(p,n,seed,nobs):
    np.random.seed(seed)
    #n_weeks = 10
    #n_hours = n_weeks * 7 * 24
    n_hours = nobs
    y = np.zeros([n_hours, 1])

    time_index = 0
    while(1):
        interarr = np.random.negative_binomial(n, p, 1)
        time_index += interarr
        if time_index < len(y):
            demand_sizes = np.random.negative_binomial(n,p, 1)
            y[time_index] = demand_sizes
        else:
            break
    return y

def convert_params(res):
    #res.params[0] = log(mu)
    #res.params[1] = alpha
    mu = np.exp(res.params[0])
    p_est = 1/(1+np.exp(res.params[0])*res.params[1])
    n_est = np.exp(res.params[0])*p_est/(1-p_est)

    return p_est , n_est

n = 3
p = 0.25
nobs = 2000
n_seeds = 20

pest_list = []
nest_list = []
for seed in range(n_seeds):
    y = generate_NBNB_data(p,n,seed,nobs)

    #split the time series into Q (interarrival times) and M (demand sizes)
    M_list = [x for x in y if x > 0]
    Q_list = []
    inter_count = 0
    for i in range(len(y)):
        if y[i] == 0:
            inter_count +=1
        else:
            Q_list.append(sum)
            inter_count = 0

    #fit distribution parameters for each sample
    X = np.ones_like(M_list)
    start_params = [1,1]
    res = sm.NegativeBinomial(M_list,X).fit(start_params)
    p_est , n_est = convert_params(res)
    pest_list.append(p_est)
    nest_list.append(n_est)
    '''
    x_plot = np.linspace(0,40,41)
    sns.set_theme()
    ax = sns.distplot(M_list, kde=False, norm_hist=True , label='Real values')
    ax.plot(x_plot, nbinom.pmf(x_plot, n_est, p_est), 'g-' , lw=2 , label='Fitted NB')
    leg = ax.legend()
    plt.title('Real vs Fitted NB Distribution for demand sizes')
    plt.show()
    '''
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
ax[0].hist(pest_list)
ax[0].axvline(p, c="red")
ax[0].set_title('Success probability')
ax[1].hist(nest_list)
ax[1].axvline(n, c="red")
ax[1].set_title('Number of successes')
plt.show()

