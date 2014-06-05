#!/usr/bin/env python
from __future__ import division
from mcmchammer import *
from scipy import stats
datafilename = 'faculty.dat'
nsamples = 10000
burn = 0
mean_candsd = 0.2
var_candsd = 0.15

# Read in Data
data = [float(line) for line in open(datafilename)]

# Use point estimators from the data to come up with starting values.
estimated_mean = stats.mean(data)
estimated_var = stats.var(data)

# Create Nodes and Links in Network
meannode = NormalNode(estimated_mean, name='Mean',
                      candsd=mean_candsd, mean=5, var=(1/3)**2)
varprior_mean = 1/4
varprior_stddev = 1/12
varprior_shape = MomentsInvGammaShape(varprior_mean,
                                      varprior_stddev**2)
varprior_scale = MomentsInvGammaScale(varprior_mean,
                                      varprior_stddev**2)
varnode = InvGammaNode(estimated_var, name='Variance',
                       candsd=var_candsd, shape=varprior_shape,
                       scale=varprior_scale)
for datum in data:
    NormalNode(datum, observed=True, mean=meannode,
               var=varnode)

# Perform simulations and plot results

currentnetwork.simulate(nsamples, burn)
meannode.plotmixing()
varnode.plotmixing()
meannode.plotmarginal(min=4, max=6.5)
varnode.plotmarginal(min=0)