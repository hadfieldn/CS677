#!/usr/bin/env python
import numpy
from node_normal import *
from node_invgamma import *
from network import *

datafilename = 'faculty.dat'
nsamples = 10000
burn = 0
mean_candsd = 0.2
var_candsd = 0.15

# Read in Data
data = [float(line) for line in open(datafilename)]

# Use point estimators from the data to come up with starting values.
estimated_mean = numpy.mean(data)
estimated_var = numpy.var(data)

def MomentsInvGammaShape(mean, var):
    return 1

def MomentsInvGammaScale(mean,var):
    return 1

# Create Nodes and Links in Network
meannode = NormalNode(estimated_mean, name='Mean', cand_var=mean_candsd, mean=5, var=(1/3)**2)
varprior_mean = 1/4
varprior_stddev = 1/12
varprior_shape = MomentsInvGammaShape(varprior_mean, varprior_stddev**2)
varprior_scale = MomentsInvGammaScale(varprior_mean, varprior_stddev**2)
varnode = InvGammaNode(estimated_var, name='Variance', cand_var=var_candsd, shape=varprior_shape, scale=varprior_scale)
for datum in data:
    NormalNode(datum, observed=True, mean=meannode, var=varnode)

# Perform simulations and plot results
network = Network([meannode, varnode])
samples = network.collect_samples(burn, nsamples)


def mean_prior_pdf(x):
    return stats.norm.pdf(x, 5, 1/3)

def var_prior_pdf(x):
    return stats.invgamma.pdf(x, a=varprior_shape, scale=varprior_scale)

prior_pdfs = { meannode: mean_prior_pdf, varnode: var_prior_pdf }

results = {}
for node in [meannode, varnode]:
    params = {
        'mean': numpy.mean(samples.of_node(node)),
        'var': numpy.var(samples.of_node(node))
    }
    results[node] = params

    title = "{}: mean = {}, var = {} (burn={}, n={})".format(node.pdf_name, params['mean'], params['var'],
                                                             burn, nsamples - burn)
    samples.plot_node(node, title=title)
    if params['var'] > 0:         # histogram fails if all values are the same
        samples.plot_histogram_for_node(node, title=title, prior_pdf=prior_pdfs[node])


