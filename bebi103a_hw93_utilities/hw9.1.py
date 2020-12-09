#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Colab setup ------------------
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade colorcet bebi103 watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../data/"
# ------------------------------

import warnings

import numpy as np
import pandas as pd
import numba

import multiprocess

import scipy.stats as st
import scipy

import iqplot
import bebi103

import tqdm.notebook as tq

import bokeh.io
import holoviews as hv
hv.extension('bokeh')

bebi103.hv.set_defaults()


# ## 

# ## 9.1a Exploratory Data Analysis

# In[10]:


# Read in data
df = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', comment='#')

# Data wrangling: melt into tidy format
df = df.melt(var_name='concentration',value_name='time to catastrophe (s)')

# Remove nan rows due to original shape of the csv
df = df.dropna()

df.head(3)


# In[11]:


# Category ordering, takes unique values of categories, sorts them, and puts them back into a string.
cat_order = [str(val) + ' uM' for val in np.sort(
    df['concentration'].apply(
        lambda x: int(x.split(' ')[0])
    ).unique()
)]

p=iqplot.ecdf(
    data=df,
    cats='concentration',
    q='time to catastrophe (s)',
    order=cat_order,
    palette=list(bokeh.palettes.Cividis5),
    tooltips=[('concentration', '@concentration'), ('TTC', '@{time to catastrophe (s)}'), ],
    plot_width=600, plot_height=480
)

bokeh.io.show(p)


# From this plotting of the various ECDF traces for microtubule catastrophe at different concentrations, we observe that there is an alleged direct relationship between microtubule catastrophe times and tubulin concentration. It appears that as tubulin concentration increases, the time to catastrophe increases. However, there are deviations in this observation, especially in the range of 400-800 seconds, and so this relationship is not exactly clearcut. Therefore, wee will perform MLE parameter estimates for under the gamma distribution generative model as well as the two successive Poisson processes with different $\beta$ generative model and observe how the MLE parameters vary with concentration. We will also perform concomitant model assessment of the two models at hand.

# ## 9.1b
# ### Model comparison using 12 uM tubulin between Gamma and a two successive Poisson processes model (two-rate gamma)
# 
# #### MLE parameter estimation for gamma distribution model
# 
# - Write generative distribution, which has the PDF:
# 
# $$
# f(t;\alpha, \beta) = \frac{1}{\Gamma(\alpha)}\frac{(\beta t)^\alpha}{t} e^{-\beta t}
# $$
# 
# t is the time to catastrophe
# 
# - Write log likelihood:
# 
# $$
# \ell(\alpha, \beta; t) = -\ln \Gamma(\alpha) + \alpha \ln (\beta t) - \ln t - \beta t
# $$
# 
# This is what I believe is implemented in `st.gamma.logpdf`
# 
# 
# - Find parameters to maximize log likelihood
#  - Substep: make a good estimate of alpha and beta based on the relationship between the parameters and the moments. (i.e., plug in estimates):
# $$
# mean: \alpha / \beta, variance: \alpha / \beta^2
# $$
#  
# $$
# \beta = \frac{mean}{variance}
# $$
# 
# $$
# \alpha = mean \times \beta = \frac{mean^2}{variance}
# $$

# In[12]:


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


# In[13]:


# Test run MLE parameter inference on 12 uM tubulin catastrophe data
tubulin_12_um = df.loc[df['concentration']=='12 uM',['time to catastrophe (s)']].values.flatten()

alpha_12_um, beta_12_um = mle_gamma(tubulin_12_um)
print("𝛼 =", alpha_12_um, "𝛽 =", beta_12_um)


# In[14]:


# Parametric bootstrap for confidence intervals 
# function to draw samples from the gamma distribution (Raymond converted this from nonparametric to parameteric)

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


# In[15]:


# try the parametric MLE function on the 12 uM tubulin catastrophe data
parametric_bs_parameters_gamma = get_bs_sample_parametric_gamma(tubulin_12_um)


# With these functions for MLE parameter estimation for the gamma distribution written. We move on to the 'two-rate' gamma distribution.

# ### Two-rate gamma distributions
# #### MLE parameter estimation for two successive Poisson processes distribution model
# 
# 
# - Write generative distribution
# 
# $$
# f(t;\beta_1, \beta_2) = \frac{\beta_1\beta_2}{\beta_2 - \beta_1}\left(\mathrm{e}^{-\beta_1 t} - \mathrm{e}^{-\beta_2 t}\right)
# $$
# 
# - Write log likelihood 
# 
# $$
# \ell(\beta_1, \beta_2, \Delta\beta; y) = \ln\beta_1 + \ln\beta_2-\ln\Delta\beta-\beta_1 t
# +\ln\left( 1-e^{-\Delta\beta t} \right)
# $$
# 
# with $\Delta\beta = \beta_2-\beta_1$.
# 
# Justin also mentioned that you can put this in terms of just $\beta_1$ and $\Delta\beta$, and this is shown below. However, we proceeded with our analyses using all three variables above.
# 
# $$
# \ell(\beta_1, \Delta\beta; y) = \ln\beta_1 - \beta_1 t - \ln\left(1 - \frac{\beta_1}{\Delta\beta}\right) + \ln\left(1 - e^{-\Delta\beta t}\right)
# $$
# 
# - Find parameters to maximize log likelihood

# In[16]:


# Get Log Likelihood of two-rate gamma
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


# In[17]:


# run MLE parameter estimation on the microtubule catastrophe data
beta_1_12_um, beta_2_12_um = mle_two_rate_gamma(tubulin_12_um)
print("𝛽1 =", beta_1_12_um, "𝛽2 =", beta_2_12_um)


# In[18]:


# Parametric bootstrap for confidence intervals 

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


# In[19]:


# try the parametric MLE function
parametric_bs_parameters_two_rate_gamma = get_bs_sample_parametric_two_rate_gamma(tubulin_12_um)


# With functions for MLE parameter estimation and parametric bootstrap for the two-gamma distribution generative model, we set out to compare the two models.

# We do a quick graphical model assessment with a predictive ECDF, a qqplot, and a CDF difference plot with the gamma distribution generative model

# In[20]:


# draw simulated catastrophe time data from the gamma distirbution with the MLE parameters above
rg = np.random.default_rng()

def draw_gamma_samples(alpha, beta, size):
    time_samples = np.empty(size)
    𝛼 = alpha
    𝛽 = beta
    
    for i in range(size):
        time_samples[i] = rg.gamma(𝛼, 1 / 𝛽)
        
    return time_samples

gamma_samples = np.array(
    [draw_gamma_samples(alpha_12_um, beta_12_um, size=len(tubulin_12_um)) for _ in range(1000)]
)

p1 = bebi103.viz.qqplot(
    data=tubulin_12_um,
    samples=gamma_samples,
    x_axis_label="time to catastrophe (s)",
    y_axis_label="time to catastrophe (s)",
)

p2 = bebi103.viz.predictive_ecdf(
    samples=gamma_samples, 
    data=tubulin_12_um, 
    discrete=True, 
    x_axis_label="time to catastrophe (s)", 
    title="Gamma distribution graphical model assessment"
)

p3 = bebi103.viz.predictive_ecdf(
    samples=gamma_samples, 
    data=tubulin_12_um, 
    diff='ecdf', 
    discrete=True, 
    x_axis_label="time to catastrophe (s)"
)

bokeh.io.show(bokeh.layouts.row(p1, p2, p3))


# We observe that there are slight deviations between the theoretical (predictive) CDF and the ECDF. This is manifested in both the predictive ECDF and the ECDF difference, which shows deviations of the data from the model from 200-400 seconds (deviations below the lower confidence interval of the simulated data) and deviations of the data from the model from 600-800 seconds (deviations above the upper confidence interval of the simulated data). slight deviations between the data and model are also seen in the qqplot from 1100 to 1500 seconds, which is recapitulated in the ECDF difference.

# We repeat the graphical model assessment with the two successive Poisson process distribution generative model (two gamma distribution)

# In[21]:


# draw simulated catastrophe time data from the gamma distirbution with the MLE parameters above
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

p1 = bebi103.viz.qqplot(
    data=tubulin_12_um,
    samples=two_rate_gamma_samples,
    x_axis_label="time to catastrophe (s)",
    y_axis_label="time to catastrophe (s)",
)

p2 = bebi103.viz.predictive_ecdf(
    samples=two_rate_gamma_samples, 
    data=tubulin_12_um, 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="Two successive Poisson process model assessment"
)

p3 = bebi103.viz.predictive_ecdf(
    samples=two_rate_gamma_samples, 
    data=tubulin_12_um, 
    diff='ecdf', 
    discrete=True, 
    x_axis_label="time to catastrophe (s)"
)

bokeh.io.show(bokeh.layouts.row(p1, p2, p3))


# Compared to the gamma distribution generative model, the two rate successive Poisson process model demonstrate larger deviations in the ECDF and the difference between the ECDFs. We will then use the gamma distribution to model the remaining data and perform MLE parameter estimates.

# # Calculating the AICs for the two models with 12 uM tubulin data

# In[22]:


model_assessment_series = pd.Series(
    index=["𝛼", "𝛽", "𝛽1", "𝛽2"],
    data=np.array([alpha_12_um, beta_12_um, beta_1_12_um, beta_2_12_um])
)

model_assessment_series["log_like_gamma"] = log_likelihood_iid_gamma(
    model_assessment_series[["𝛼", "𝛽"]],
    tubulin_12_um
)

model_assessment_series["log_like_two_rate_gamma"] = log_likelihood_iid_two_rate_gamma(
    model_assessment_series[["𝛽1", "𝛽2"]],
    tubulin_12_um
)

model_assessment_series["AIC_gamma"] = -2 * (model_assessment_series['log_like_gamma'] - 2)
model_assessment_series["AIC_two_rate_gamma"] = -2 * (model_assessment_series['log_like_two_rate_gamma'] - 2)

AIC_max = max(model_assessment_series[['AIC_gamma', 'AIC_two_rate_gamma']])
numerator = np.exp(-(model_assessment_series.loc['AIC_gamma'] - AIC_max)/2)
denominator = numerator + np.exp(-(model_assessment_series['AIC_two_rate_gamma'] - AIC_max)/2)
model_assessment_series['w_gamma'] = numerator / denominator
model_assessment_series['w_two_rate_gamma'] = 1 - model_assessment_series['w_gamma']

model_assessment_series


# From the AICs and weights of the two models, we see that the gamma distribution generative model is favored over the two successive Poisson processes model, this corroborates our graphical model assessment above.

# ## 9.1c Calculating MLE parameters for microtubule catastrophe time data at all concentrations of tubulin using the gamma distribution

# In[27]:


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


# In[28]:


# generate a dictionary of alpha and beta values for the gamma distribution for catastrophe data at each tubulin concentration
MLE_parameters_dict = {
    concentration: mle_gamma(
        df.loc[df['concentration'] == concentration, 'time to catastrophe (s)'].values.flatten()
    ) for concentration in tq.tqdm(df['concentration'].unique(), unit='concentrations')
}

MLE_parameters_dict


# In[29]:


# Draw parametric bootstrap replicates for the MLE parameters and calculate confidence intervals from the alpha and beta value dictionary
parametric_bootstrap_replicates_dict = {
    concentration : get_bs_sample_parametric_gamma(
        df.loc[df['concentration'] == concentration, 'time to catastrophe (s)'].values.flatten()
    )[0] for concentration in tq.tqdm(df['concentration'].unique(), unit='concentrations')
}


# In[30]:


for conc in cat_order:
    # Current 2D array of replicates
    parameter_replicates = parametric_bootstrap_replicates_dict[conc]
    
    # Generate summary dictionaries
    β_dict = gen_summaries_gamma(parameter_replicates)

    # Just about ready to conc out
    bokeh.io.show(bebi103.viz.confints(β_dict, title=conc))
    
    # change this to alpha and beta, change into two confidence interval diagrams with all the alphas and betas from each curve split


# ### Plot MLE parameters and the bootstrap replicates

# In[31]:


# Put bootstrap replicates into near-tidy dataframe
replicate_df = pd.DataFrame([[k, v] for k, v in parametric_bootstrap_replicates_dict.items()], columns=['conc', 'params'])  

# Sort for posterity. First, add 
replicate_df['conc (int)'] = np.vstack(replicate_df['conc'].str.split(' '))[:,0].astype(int)

replicate_df.head()


# In[32]:


replicate_df = replicate_df.explode('params')
replicate_df.head()


# In[33]:


# First, we take the whole betas column and put it into a 2d array (again. one step forward two steps in one of four diagonal directions. Where am I going? Who are you?)
params = np.vstack(replicate_df.params.values)

# Now, individually assign to each column and remove redundant beta column.
replicate_df['𝛼'], replicate_df['β'] = params[:,0], params[:,1]
del replicate_df['params']
replicate_df.head()


# In[34]:


# repeat this data wrangling for MLE parameters

# Put into near-tidy dataframe
gamma_df = pd.DataFrame([[k, v] for k, v in MLE_parameters_dict.items()], columns=['conc', 'params'])  

# Sort for posterity. First, add 
gamma_df['conc (int)'] = np.vstack(gamma_df['conc'].str.split(' '))[:,0].astype(int)

# First, we take the whole betas column and put it into a 2d array (again. one step forward two steps in one of four diagonal directions. Where am I going? Who are you?)
params = np.vstack(gamma_df.params.values)

# Now, individually assign to each column and remove redundant beta column.
gamma_df['𝛼'], gamma_df['β'] = params[:,0], params[:,1]
del gamma_df['params']
gamma_df.head()


# In[35]:


# visualize bootstrap replicates and MLE parameters for each concentration
replicate_scatter = hv.Points(
    data=replicate_df,
    kdims=['𝛼','β'],
    vdims=['conc'],
    label='parameter values',
).groupby(
    'conc',
).opts(
    fill_alpha=0.7,
    line_alpha=0,
    size=2,
    frame_height=480,
    frame_width=600,
).overlay()

mle_scatter = hv.Points(
    data=gamma_df,
    kdims=['𝛼','β'],
    vdims=['conc'],
    label='parameter values',
).groupby(
    'conc',
).opts(
    fill_alpha=0.7,
    line_alpha=0,
    size=8,
    frame_height=480,
    frame_width=600,
).overlay()

replicate_scatter * mle_scatter


# In this plot of the values of the parameters 𝛼 and β for the gamma distribution generative model of microtubule catastrophe, we observe an direct relationship between the value of beta and the value of alpha within a microtubule concentration. Across different tubulin concentrations, we observe an inverse correlation of β with tubulin concentration and a direct correlation of 𝛼 with tubulin concentration. This relationship does not hold with very low tubulin concentration, such as 7 um (red bootstrap replicates), where it seems to have outlier β values (lower than expected). This suggests that the gamma distribution model could be a less accurate description of generative model for microtubule catastrophe at low tubulin concentrations compared to higher tubulin concentrations. This warrants a cursory graphical model assessment and AIC weight calculation over all tubulin concentrations, which we will carry out below.

# ### Attempt to recapitulate ECDF of catastrophe data at each tubulin concentration with the theoretical CDFs from MLE parameters --> graphical model assessment 

# In[36]:


# add number of replicates to the gamma parameter dataframe

tubulin_14_um = df.loc[df['concentration']=='14 uM',['time to catastrophe (s)']].values.flatten()
tubulin_12_um = df.loc[df['concentration']=='12 uM',['time to catastrophe (s)']].values.flatten()
tubulin_10_um = df.loc[df['concentration']=='10 uM',['time to catastrophe (s)']].values.flatten()
tubulin_9_um = df.loc[df['concentration']=='9 uM',['time to catastrophe (s)']].values.flatten()
tubulin_7_um = df.loc[df['concentration']=='7 uM',['time to catastrophe (s)']].values.flatten()

data_list = [tubulin_12_um, tubulin_7_um, tubulin_9_um, tubulin_10_um, tubulin_14_um]

gamma_df["dataset length"] = [len(n) for n in data_list]
gamma_df


# In[37]:


# dictionary of catastrophe data simulated by the gamma distribution with the proper parameters 

gamma_dict = {
    concentration: np.array([draw_gamma_samples(
        float(gamma_df.loc[gamma_df["conc"] == concentration, "𝛼"]), 
        float(gamma_df.loc[gamma_df["conc"] == concentration, "β"]), 
        size=int(gamma_df.loc[gamma_df["conc"] == concentration, "dataset length"])
    ) for _ in range (1000)]) for concentration in gamma_df['conc'].unique()
}


# In[38]:


s1 = bebi103.viz.qqplot(
    data=tubulin_7_um,
    samples=gamma_dict["7 uM"],
    x_axis_label="time to catastrophe (s)",
    y_axis_label="time to catastrophe (s)",
    title="7 uM"
)

s2 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["7 uM"], 
    data=tubulin_7_um, 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="7 uM"
)

s3 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["7 uM"], 
    data=tubulin_7_um, 
    diff='ecdf', 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="7 uM"
)

s4 = bebi103.viz.qqplot(
    data=tubulin_9_um,
    samples=gamma_dict["9 uM"],
    x_axis_label="time to catastrophe (s)",
    y_axis_label="time to catastrophe (s)",
    title="9 uM"
)

s5 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["9 uM"], 
    data=tubulin_9_um, 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="9 uM"
)

s6 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["9 uM"], 
    data=tubulin_9_um, 
    diff='ecdf', 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="9 uM"
)

s7 = bebi103.viz.qqplot(
    data=tubulin_10_um,
    samples=gamma_dict["10 uM"],
    x_axis_label="time to catastrophe (s)",
    y_axis_label="time to catastrophe (s)",
    title="10 uM"
)

s8 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["10 uM"], 
    data=tubulin_10_um, 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="10 uM"
)

s9 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["10 uM"], 
    data=tubulin_10_um, 
    diff='ecdf', 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="10 uM"
)

s10 = bebi103.viz.qqplot(
    data=tubulin_14_um,
    samples=gamma_dict["14 uM"],
    x_axis_label="time to catastrophe (s)",
    y_axis_label="time to catastrophe (s)",
    title="14 uM"
)

s11 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["14 uM"], 
    data=tubulin_14_um, 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="14 uM"
)

s12 = bebi103.viz.predictive_ecdf(
    samples=gamma_dict["14 uM"], 
    data=tubulin_14_um, 
    diff='ecdf', 
    discrete=True, 
    x_axis_label="time to catastrophe (s)",
    title="14 uM"
)

grid = bokeh.layouts.gridplot([[s1, s2, s3], [s4, s5, s6], [s7, s8, s9], [s10, s11, s12]], plot_width=250, plot_height=250)

bokeh.io.show(grid)


# AHA! Out of all the datasets, the 7 uM tubulin microtubule catastrophe data is the least commensurate with the gamma distribution generative model, consistent with our predictions.

# ## Bonus: AIC weight calculations for the two models at other tubulin concentrations

# In[39]:


model_assessment_mle = pd.DataFrame(index=['alpha', 'beta', 'beta1', 'beta2', 
                                           "log_like_gamma", "log_like_two_rate_gamma", 
                                           "AIC_gamma", "AIC_two_rate_gamma", 
                                           'w_gamma', 'w_two_rate_gamma'])

for tubulin in df['concentration'].unique():
    
    data = df.loc[df['concentration']==tubulin,['time to catastrophe (s)']].values.flatten()

    # gamma MLE
    alpha, beta = mle_gamma(data)

    # two-rate gamma MLE
    beta1, beta2 = mle_two_rate_gamma(data)
    
    # gamma log likelihood 
    gamma_lle = log_likelihood_iid_gamma(
        (alpha, beta), 
        data
    )
    
    # two-rate gamma log likelihood
    two_rate_lle = log_likelihood_iid_two_rate_gamma(
        (beta1, beta2), 
        data
    )
    
    # AIC for gamma
    gamma_aic = -2 * (gamma_lle - 2)
    
    # AIC for two-rate gamma
    two_rate_aic = -2 * (two_rate_lle - 2)
    
    # AIC weight for gammma
    AIC_max = max(gamma_aic, two_rate_aic)
    numerator = np.exp(-(gamma_aic - AIC_max)/2)
    denominator = numerator + np.exp(-(two_rate_aic - AIC_max)/2)    
    
    gamma_weight = numerator / denominator
    
    # AIC weight for two-rate gamma
    two_rate_weight = 1 - gamma_weight
    
    # Store results in data frame
    model_assessment_mle[tubulin] = [alpha, beta, beta1, beta2, gamma_lle, two_rate_lle, gamma_aic, two_rate_aic, gamma_weight, two_rate_weight]

# Take a look
model_assessment_mle


# The AIC weights still predict that the gamma distribution is the best generative model for the data, even at low concentrations, but we see that the weight for the gamma distribution is slightly lower at lower concentrations.

# **Note for the analysis below**: Our group also performed MLE parameter estimation and confidence intervals for catastrophe data taken at all concentrations of tubulin using the two-rate gamma model (two successive Poisson processes). The confidence intervals were computed with bootstrap replicates drawn with both nonparametric, and parametric bootstrap methods. Each approach generated different structure in the bootstrap replicates, which has led to interesting insights on the generative model behind the microtubule catastrophe process.

# ## Calculating MLE parameters and confidence intervals for the selected model for all concentrations of tubulin using nonparametric bootstrap using the two-rate gamma model

# In[40]:


# this is nonparametric version
def nonparametric_two_rate_bs_MLE(data, n_replicates):
    # Bootstrap for parameters
    return np.vstack([
        mle_two_rate_gamma(
            get_bs_sample_nonparametric(data)
        ) for _ in tq.tqdm(range(n_replicates), unit='replicates')
    ])


n_replicates = 1000
arg_iterable = {
    concentration: nonparametric_two_rate_bs_MLE(
        df.loc[df['concentration']==concentration,'time to catastrophe (s)'].values,
        n_replicates
    ) for concentration in tq.tqdm(df['concentration'].unique(), unit='concentrations')
} # needs to be a list of tuples


# ## Confidence intervals for two-rate gamma model by nonparametric bootstrap for parameter values of the two-rate gamma model

# In[41]:


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


# In[42]:


# Using ordered list of concentrations from above
# visualization of nonparametric bootstrap confidence intervals
for conc in cat_order:
    # Current 2D array of replicates
    parameter_replicates = arg_iterable[conc]
    
    # Generate summary dictionaries
    β_dict = gen_summaries(parameter_replicates)

    # Just about ready to conc out
    bokeh.io.show(bebi103.viz.confints(β_dict, title=conc))


# ## Look at distribution of beta across distributions for the two-rate gamma model
# ### First, wrangle arg_iterable back into a dataframe. Surprisingly tricky.

# In[43]:


# Put into near-tidy dataframe
beta_df = pd.DataFrame([[k, v] for k, v in arg_iterable.items()], columns=['conc', 'betas'])  

# Sort for posterity. First, add 
beta_df['conc (int)'] = np.vstack(beta_df['conc'].str.split(' '))[:,0].astype(int)

beta_df.head()


# Right now, each betas cell has the entire column. We want to separate them into their own columns

# In[44]:


beta_df=beta_df.explode('betas')
beta_df.head()


# We are close, but we want the betas to be in two different rows.
# 
# Solution for splitting from https://www.geeksforgeeks.org/split-a-text-column-into-two-columns-in-pandas-dataframe/

# In[45]:


# First, we take the whole betas column and put it into a 2d array (again. one step forward two steps in one of four diagonal directions. Where am I going? Who are you?)
betas = np.vstack(beta_df.betas.values)

# Now, individually assign to each column and remove redundant beta column.
beta_df['β1'], beta_df['β2'] = betas[:,0], betas[:,1]
del beta_df['betas']
beta_df.head()


# In[46]:


hv.Points(
    data=beta_df,
    kdims=['β1','β2'],
    vdims=['conc'],
    label='β values',
).groupby(
    'conc',
).opts(
    fill_alpha=0.7,
    line_alpha=0,
    size=2,
    frame_height=480,
    frame_width=600,
).overlay()


# Some strange things are happening here. It looks like at lower concentrations (7 $\mu$M), it is best for $\beta_2$ to be higher than $\beta_1$. Once the concentraton is at some threshold around 9 $\mu$M, $\beta_1$ and $\beta_2$ start to look identical. However, deviations from this pattern again push $\beta_2$ above $\beta_1$.

# ### Parametric bootstrap of the MLE parameters for the two-rate gamma model

# In[47]:


# parametric bootstrap equivalent to the one above

two_rate_parametric_bootstrap_replicates_dict = {
    concentration : get_bs_sample_parametric_two_rate_gamma(
        df.loc[df['concentration'] == concentration, 'time to catastrophe (s)'].values.flatten()
    )[0] for concentration in tq.tqdm(df['concentration'].unique(), unit='concentration')
}


# In[48]:


# Using ordered list of concentrations from above
# visualize these parametric bootstrap confidence intervals
for conc in cat_order:
    # Current 2D array of replicates
    parameter_replicates = two_rate_parametric_bootstrap_replicates_dict[conc]
    
    # Generate summary dictionaries
    β_dict = gen_summaries(parameter_replicates)

    # Just about ready to conc out
    bokeh.io.show(bebi103.viz.confints(β_dict, title=conc))


# In[49]:


# Put into near-tidy dataframe
two_rate_replicate_df = pd.DataFrame([[k, v] for k, v in two_rate_parametric_bootstrap_replicates_dict.items()], columns=['conc', 'betas'])  

# Sort for posterity. First, add 
two_rate_replicate_df['conc (int)'] = np.vstack(two_rate_replicate_df['conc'].str.split(' '))[:,0].astype(int)

two_rate_replicate_df = two_rate_replicate_df.explode('betas')

# First, we take the whole betas column and put it into a 2d array (again. one step forward two steps in one of four diagonal directions. Where am I going? Who are you?)
betas = np.vstack(two_rate_replicate_df.betas.values)

# Now, individually assign to each column and remove redundant beta column.
two_rate_replicate_df['β1'], two_rate_replicate_df['β2'] = betas[:,0], betas[:,1]
del two_rate_replicate_df['betas']


# In[50]:


# visualize the parametric boostrap replicates for the two-rate gamma model

hv.Points(
    data=two_rate_replicate_df,
    kdims=['β1','β2'],
    vdims=['conc'],
    label='beta values',
).groupby(
    'conc',
).opts(
    fill_alpha=0.7,
    line_alpha=0,
    size=2,
    frame_height=480,
    frame_width=600,
).overlay()


# The parametric bootstrap replicates of the MLE parameters for the two successive Poisson processes model demonstrate interesting structure in which there are large variations of β2 over all tubulin concentrations above the line where β1 = β2. This structure is not generally recapitulated in the plot of the nonparametric bootstrap replicates, and only with the 7 uM tubulin concentration bootstrap replicates, suggesting that the two successive Poisson processes model does not generate bootstrap replicates that is consistent with bootstrap replicates drawn from the data itself (nonparametrically) other than the 7 uM tubulin condition. This could be additional support for the fact that the gamma distribution is a better generative model than the two successive Poisson processes model over all tubulin concentration except at lower concentrations of tubulin. There could be a biochemical basis to this observation, in that at low tubulin concentrations there is a change in the mechanism or kinetics of the catastrophe process that leads to data that could be better modeled with the two successive Poisson processes model.

# ## Attribution

# Kate, Raymond, and Dan wrote the code.

# In[51]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p numpy,scipy,pandas,numba,bebi103,iqplot,bokeh,holoviews,tqdm,jupyterlab')

