import warnings

import numpy as np
import pandas as pd
import numba

import multiprocess

import scipy.stats as st
import scipy

import iqplot
import bebi103

import tqdm as tq

import bokeh.io
import holoviews as hv
hv.extension('bokeh')

bebi103.hv.set_defaults()

# %%
# Get Log Likelihood of gamma
@numba.njit
def get_sum(vals):
    return np.sum(vals)


def log_likelihood_iid_gamma(params, data):
    '''
    Generates sum of log likelihood values
    for each value in data with parameters params
    
    parameters
    ----------
    params (array-like) : array of parameters 𝛼 and 𝛽 for parameterizing the gamm distribution
    
    data (array-like) : array of microtubule catastrophe times in seconds 
    
    returns
    -------
    (float) : the value of the log likelihood function 
    
    '''
    
    𝛼, 𝛽 = params
    if 𝛼 <= 0 or 𝛽 <= 0:
        return -np.inf
    
    # Note: scipy parameterizes gamma with 𝛼 and b = 1/𝛽
    return np.sum(st.gamma.logpdf(data, 𝛼, loc=0, scale=1/𝛽))


@numba.njit
def get_initial_parameters_gamma(data):
    '''
    Uses moments of data to estimate mean and variances
    
    parameters
    ----------
    data (array-like) : array of microtubule catastrophe times in seconds 
    
    returns
    -------
    (tuple) : initial guesses for parameters 𝛼 and 𝛽
    
    '''
    curr_mean = np.mean(data)
    variance = np.var(data)
    
    𝛽 = curr_mean / variance
    𝛼 = curr_mean**2 / variance
    
    return 𝛼, 𝛽

# optimization routine for the log likelihood 
def mle_gamma(data):
    ''' 
    Parameters
    ----------
    data: np.array
        1d array of utubule catastrophe times in seocnds 
        
    Return
    ------
    res: (dict)
        res.x is a tuple, (alpha, beta)
    
    Notes
    -----
    params: (tuple)
        Initial guess for alpha and beta
    '''
    
    initial_guess = get_initial_parameters_gamma(data)
    
    with warnings.catch_warnings():
        # Oh hell yes
        warnings.simplefilter("ignore")
        
        res = scipy.optimize.minimize(
            fun=lambda params, data: -log_likelihood_iid_gamma(
                params,
                data
            ),
            x0=initial_guess,
            args=(data,), # data is like "training data"
            method='powell',
            tol=1e-7
        )
    
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)



def gen_gamma(params, size, rg):
    """ 
    generate bootstrap samples of microtubule catastrophe data for use with bebi103.bootstrap.draw_bs_reps_mle()
    
    parameters
    ----------
    params (array-like) : alpha and beta for parametrized the gamma distribution
    
    size (int) : number of values to draw from the distribution
    
    rg (generator) : name of random number generatory (np.random.default_rng())
    
    returns
    -------
    np.array of the values drawn out of the gamma distribution parametrized by params
    
    """
    
    𝛼, 𝛽 = params
    return rg.gamma(𝛼, 1 / 𝛽, size=size)

def get_bs_sample_parametric_gamma(data, size=1000, n_jobs=1, progress_bar=False):
    """
    generate parametric bootstrap replicates of the MLE parameters for the gamma distribution
    
    parameters
    ----------
    data (array-like) : microtubule catastrophe time data in seconds
    
    returns
    -------
    tuple of MLE parameter bootstrap replicates and computed confidence intervals
    """
    
    # draw boostrap replicates for the parameters of the gamma distribution model
    bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
        mle_fun=mle_gamma,
        gen_fun=gen_gamma,
        data=data,
        size=size,
        n_jobs=n_jobs,
        progress_bar=progress_bar
    )
    
    # compute confidnece intervals
    conf_ints = np.percentile(bs_reps, [2.5, 97.5], axis=0)
    
    # Print the results
    print(
        """95% Confidence intervals
    𝛼: [ {0:.4f},  {1:.4f}]
    𝛽: [{2:.4f}, {3:.4f}]
    """.format(
            *conf_ints.T.ravel()
        )
    )
    
    return bs_reps, conf_ints


def log_likelihood_iid_two_rate_gamma(params, t):
    '''
    Generates sum of log likelihood values for a two-rate gamma distribution
    for each value in data with parameters params
    
    I think we can assume this is not a gamma distribution, otherwise why would we be modeling this
    
    Parameters
    ----------
    params (array-like) : the values of beta1 and beta2 parametrizing the generative model
    
    t (array-like) : microtubule catastrophe time data in seconds
    
    Returns
    -------
    the sum of the log likelihood function with the given parameters and data
    '''
    
    𝛽_1, 𝛽_2 = params
    d𝛽 = 𝛽_2 - 𝛽_1
    
    if 𝛽_1 <= 0 or 𝛽_2 <= 0:
        # If params are negative, that's a no go
        return -np.inf
    elif d𝛽 == 0:
        # If 𝛽_1 equal to 𝛽_2, then negative infinity
        return -np.inf
    elif 𝛽_1 > 𝛽_2:
        # Avoiding indistinguishability: beta1 must be less than beta2
        return -np.inf
    elif np.isclose(𝛽_1, 𝛽_2):
        # If 𝛽_1 is too close to 𝛽_2, then it is approximating a gamma distribution.
        return get_sum(st.gamma.logpdf(t, 2, loc=0, scale=1/𝛽_1)) # alpha = 2 because two gamma processes    
    
    p1 = np.log(𝛽_1)
    # p2 = - np.log(1 - 𝛽_1/𝛽_2) 
    p2 = np.log(1 + (𝛽_1/ d𝛽)) # unsure if this is correct np.log(1 + (𝛽_1/ d𝛽))
    p3 = - 𝛽_1 * t
    p4 = np.log(1 - np.exp(-d𝛽 * t))
    
    # ll is log likelihood
    ll = p1 + p2 + p3 + p4

    # Get sum
    ll_sum = get_sum(ll)

    return ll_sum


def mle_two_rate_gamma(data):
    ''' 
    Runs optimization of the log likelihood function with Powell's method to converge on the MLE parameters 
    
    Parameters
    ----------
    data: np.array
        1d array of utubule catastrophe times
        
    Return
    ------
    res: (dict)
        res.x is a tuple, (alpha, beta)
    
    Notes
    -----
    params: (tuple)
        Initial guess for alpha and beta'''
    
    initial_params = (0.025,0.05) # 𝛽_1 < 𝛽_2
    
    with warnings.catch_warnings():
        # Oh hell yes
        warnings.simplefilter("ignore")
        
        res = scipy.optimize.minimize(
            fun=lambda params, data: -log_likelihood_iid_two_rate_gamma(
                params,
                data
            ),
            x0=initial_params,
            args=(data,), # data is like "training data"
            method='powell',
            tol=1e-7
        )
    
    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

def gen_two_rate_gamma(params, size, rg):
    """ 
    generate bootstrap samples of microtubule catastrophe data for use with bebi103.bootstrap.draw_bs_reps_mle()
    
    parameters
    ----------
    params (array-like) : alpha and beta for parametrized the two rate gamma distribution
    
    size (int) : number of values to draw from the distribution
    
    rg (generator) : name of random number generator (np.random.default_rng())
    
    returns
    -------
    np.array of the values drawn out of the two rate gamma distribution parametrized by params
    
    """

    𝛽1, 𝛽2 = params
    return rg.exponential(1 / 𝛽1, size=size) + rg.exponential(1 / 𝛽2, size=size)

# @numba.njit
def get_bs_sample_parametric_two_rate_gamma(data, size=1000, n_jobs=1, progress_bar=False):
    """
    generate parametric bootstrap replicates of the MLE parameters for the two rate gamma distribution
    
    parameters
    ----------
    data (array-like) : microtubule catastrophe time data in seconds
    
    returns
    -------
    tuple of MLE parameter bootstrap replicates and computed confidence intervals
    """
    
    # draw boostrap replicates for the parameters of the gamma distribution model
    bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
        mle_fun=mle_two_rate_gamma,
        gen_fun=gen_two_rate_gamma,
        data=data,
        size=size,
        n_jobs=n_jobs,
        progress_bar=progress_bar
    )
    
    # compute confidnece intervals
    conf_ints = np.percentile(bs_reps, [2.5, 97.5], axis=0)
    
    # Print the results
    print(
        """95% Confidence intervals
    𝛽1: [ {0:.4f},  {1:.4f}]
    𝛽2: [{2:.4f}, {3:.4f}]
    """.format(
            *conf_ints.T.ravel()
        )
    )
    
    return bs_reps, conf_ints


def draw_gamma_samples(alpha, beta, size):
    time_samples = np.empty(size)
    𝛼 = alpha
    𝛽 = beta
    
    for i in range(size):
        time_samples[i] = rg.gamma(𝛼, 1 / 𝛽)
        
    return time_samples

rg = np.random.default_rng()

def draw_two_rate_gamma_samples(beta1, beta2, size):
    time_samples = np.empty(size)
    𝛽1 = beta1
    𝛽2 = beta2
    
    for i in range(size):
        time_samples[i] = rg.exponential(1 / 𝛽1) + rg.exponential(1 / 𝛽2)
        
    return time_samples

two_rate_gamma_samples = np.array(
    [draw_two_rate_gamma_samples(beta_1_12_um, beta_2_12_um, size=len(tubulin_12_um)) for _ in range(1000)]
)


def gen_summaries_gamma(data, fun=np.mean):
    '''Generate conf_int summaries for the gamma distribution MLE parameters 
    
    Parameters
    ----------
    data: np.array
        2D numpy array containing columns of 𝛼 and β bootstrap replicates
    
    Returns
    -------
    summaries: list
        list of dictionaries conforming to format from bebi103.viz.confints() `summaries` parameter spec.
    '''
    # Initiatlize holder list
    summaries = []
    
    # for 𝛼 and β,
    for param in range(2):
        if param == 0:
            label = "𝛼"
        elif param == 1:
            label = "β"
        curr_d = data[:,param]
        
        # Create dictionary
        summaries.append({
            'estimate':fun(curr_d),
            'conf_int':np.percentile(curr_d,[2.5, 97.5]),
            'label':label
        })
    
    return summaries

def nonparametric_two_rate_bs_MLE(data, n_replicates):
    # Bootstrap for parameters
    return np.vstack([
        mle_two_rate_gamma(
            get_bs_sample_nonparametric(data)
        ) for _ in tq.tqdm(range(n_replicates), unit='replicates')
    ])

def gen_summaries(data, fun=np.mean):
    '''Generate conf_int summaries
    
    Parameters
    ----------
    data: np.array
        2D numpy array containing columns of β_1 and β_2 replicates
    
    Returns
    -------
    summaries: list
        list of dictionaries conforming to format from bebi103.viz.confints() `summaries` parameter spec.
    '''
    # Initiatlize holder list
    summaries = []
    
    # for β_1 and β_2,
    for beta in range(2):
        # Get the current column of betas
        curr_d = data[:,beta]
        
        # Create dictionary
        summaries.append({
            'estimate':fun(curr_d),
            'conf_int':np.percentile(curr_d,[2.5, 97.5]),
            'label':'β_{}'.format(beta+1)
        })
    
    return summaries