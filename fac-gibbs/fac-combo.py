#!/usr/bin/env python
"""Gibbs/Metropolis Combo example: Faculty Evaluations"""

import evilplot
import math
import random
import sys

# Number of samples and Burn
nsamples = 5000
burn = 50

priors = dict(mean_of_mean=5, var_of_mean=1/9, alpha_of_var=11, beta_of_var=2.5)
init_values = dict(mean=5, var=0.3)
candsd = dict(mean=0.2, var=0.15)

if sys.version_info < (3, 2):
    raise RuntimeError('Python 3.2 or better is required')

def gibbs_generator(priors, data, init_values, candsd):
    """The Gibbs-Metropolis MCMC infinite loop"""
    mean_value = init_values['mean']
    var_value = init_values['var']

    while True:
        mean_value = sample_mean(mean_value, priors, var_value, data, candsd['mean'])
        var_value = sample_var(var_value, priors, mean_value, data, candsd['var'])
        yield {'mean': mean_value, 'var': var_value}

def sample_mean(last, priors, var_value, data, candsd):
    """Generate a new sample of the mean variable

    Uses Metropolis with calculations on the log scale.
    """
    prior_mean = priors['mean_of_mean']
    prior_var = priors['var_of_mean']
    cand =  random.gauss(last, candsd)


    log_lh_last = mean_logconditional(last, prior_mean, prior_var)
    log_lh_cand = mean_logconditional(cand, prior_mean, prior_var)
    for y in data:
        log_lh_last += obs_logconditional(y, last, var_value)
        log_lh_cand += obs_logconditional(y, cand, var_value)

    u = math.log(random.random())

    if u < log_lh_cand - log_lh_last:
        return cand
    else:
        return last

def sample_var(last, priors, mean_value, data, candsd):
    """Generate a new sample of the variance variable

    Uses Metropolis with calculations on the log scale.
    """
    prior_alpha = priors['alpha_of_var']
    prior_beta = priors['beta_of_var']
    cand =  random.gauss(last, candsd)
    if cand <= 0:
        return last

    log_lh_last = var_logconditional(last, prior_alpha, prior_beta)
    log_lh_cand = var_logconditional(cand, prior_alpha, prior_beta)
    for y in data:
        log_lh_last += obs_logconditional(y, mean_value, last)
        log_lh_cand += obs_logconditional(y, mean_value, cand)

    u = math.log(random.random())

    if u < log_lh_cand - log_lh_last:
        return cand
    else:
        return last

def obs_logconditional(obs, mean, var):
    return -1/2 * (math.log(var) + 1/var * (obs-mean)**2)

def mean_logconditional(mu, prior_mean, prior_var):
    return -1/(2*prior_var) * (mu - prior_mean)**2

def var_logconditional(var, prior_alpha, prior_beta):
    return -(prior_alpha + 1) * math.log(var) - prior_beta / var


##############################################################################
### Generate Samples

# Read in the Data
data = [float(line) for line in open('faculty.dat')]

# Generator that will produce an infinite number of samples:
mcmc = gibbs_generator(priors, data, init_values, candsd)

# But we don't have time for an infinite number of samples.  We'll settle for
# the ones between `burn` and `burn + nsamples`
for i in range(burn):
    next(mcmc)
samples = [next(mcmc) for i in range(nsamples)]

mean_marginal = [s['mean'] for s in samples]
var_marginal = [s['var'] for s in samples]


##############################################################################
### Plots of Mixing

def plotmixing(samples, name):
    p = evilplot.Plot(title='%s mixing'%(name))
    points = evilplot.Points(list(enumerate(samples)))
    points.style = 'lines'
    points.linewidth = 1
    p.append(points)
    #p.write_gpi('plots/mix-%s.gpi' % name)
    p.show()

plotmixing(mean_marginal, 'mean')
plotmixing(var_marginal, 'var')


########################################################################
# Prior/Posterior Plots

def mean_prior_pdf(x):
    """Compute the Normal pdf at `x` given the priors."""
    mean = priors['mean_of_mean']
    var = priors['var_of_mean']
    return ((1 / (2 * math.pi * var) ** 0.5) *
            math.exp(-1 / (2 * var) * (x - mean) ** 2))

def var_prior_pdf(x):
    """Compute the Inverse Gamma pdf at `x` given the priors.

    Note that we use the scale parameterization of `beta`.
    """
    alpha = priors['alpha_of_var']
    beta = priors['beta_of_var']
    return (beta ** alpha / math.gamma(alpha) * x**(-alpha - 1)
            * math.exp(-beta / x))

def plotposterior(samples, prior_pdf, name, xmin, xmax):
    p = evilplot.Plot(title='Prior and Posterior of %s'%(name))
    p.ymin = 0.0
    p.xmin = xmin
    p.xmax = xmax

    priord = evilplot.Function(prior_pdf)
    priord.title = 'Prior Dist'
    p.append(priord)

    postd = evilplot.Histogram(samples, 30, normalize=True)
    postd.title = 'Posterior Dist'
    p.append(postd)

    #p.write_gpi('plots/post-%s.gpi' % name)
    p.show()

plotposterior(mean_marginal, mean_prior_pdf, 'mean', 5.0, 6.5)
plotposterior(var_marginal, var_prior_pdf, 'var', 0.0001, 1.0)


########################################################################
# Predictive Posterior Plot

postpred = [random.gauss(s['mean'], math.sqrt(s['var'])) for s in samples]

p = evilplot.Plot(title='Posterior Predictive Distribution')
p.ymin = 0.0
d = evilplot.Histogram(postpred, 30, normalize=True)
p.append(d)

#p.write_gpi('plots/predictive.gpi')
p.show()

# vim: et sw=4 sts=4
