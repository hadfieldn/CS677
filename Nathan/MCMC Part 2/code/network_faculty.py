#!/usr/bin/env python
import numpy
from node_normal import *
from node_invgamma import *
from network import *
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
_log = logging.getLogger("faculty")

datafilename = 'faculty.dat'
nsamples = 1000
burn = 10000
mean_candsd = 0.2
var_candsd = 0.15

# Read in Data
data = [float(line) for line in open(datafilename)]

# Use point estimators from the data to come up with starting values.
estimated_mean = numpy.mean(data)
estimated_var = numpy.var(data)

def MomentsInvGammaShape(mean, var):
    return 11


def MomentsInvGammaScale(mean,var):
    return 2.5


def mean_prior_pdf(x):
    return stats.norm.pdf(x, 5, 1/3)


def var_prior_pdf(x):
    varprior_mean = 1/4
    varprior_stddev = 1/12
    varprior_shape = MomentsInvGammaShape(varprior_mean, varprior_stddev**2)
    varprior_scale = MomentsInvGammaScale(varprior_mean, varprior_stddev**2)
    return stats.invgamma.pdf(x, a=varprior_shape, scale=varprior_scale)

varprior_mean = 1/4
varprior_stddev = 1/12
varprior_shape = MomentsInvGammaShape(varprior_mean, varprior_stddev**2)
varprior_scale = MomentsInvGammaScale(varprior_mean, varprior_stddev**2)


def nodes_for_learning_hyperparam_for_step(step, learned_nodes):
    steps = {
        0: nodes_without_learning_hyperparams,
        1: nodes_for_learning_one_hyperparam,
        2: nodes_for_learning_two_hyperparams,
        3: nodes_for_learning_three_hyperparams,
        4: nodes_for_learning_four_hyperparams
    }
    return steps[step](learned_nodes)


def nodes_without_learning_hyperparams(learned_nodes):
    mean_node = NormalNode(estimated_mean, name='Mean', cand_var=mean_candsd, mean=5, var=(1/3)**2)

    var_node = InvGammaNode(estimated_var, name='Variance', cand_var=var_candsd, shape=varprior_shape, scale=varprior_scale)

    return mean_node, var_node, None


def nodes_for_learning_one_hyperparam(learned_nodes):
    learned_node = NormalNode(0, "HyperMean-Mean", 5, 1)
    mean_node = NormalNode(estimated_mean, name='Mean', cand_var=mean_candsd, mean=learned_node, var=(1/3)**2)
    var_node = InvGammaNode(estimated_var, name='Variance', cand_var=var_candsd, shape=varprior_shape, scale=varprior_scale)
    for datum in data:
        NormalNode(datum, observed=True, mean=mean_node, var=var_node)

    return mean_node, var_node, learned_node


def nodes_for_learning_two_hyperparams(learned_nodes):
    learned_node = InvGammaNode(0.1, name='HyperMean-Variance', cand_var=var_candsd, shape=1, scale=1)

    mean_node = NormalNode(estimated_mean, name='Mean', cand_var=mean_candsd, mean=learned_nodes[0], var=learned_node)
    var_node = InvGammaNode(estimated_var, name='Variance', cand_var=var_candsd, shape=varprior_shape, scale=varprior_scale)
    for datum in data:
        NormalNode(datum, observed=True, mean=mean_node, var=var_node)

    return mean_node, var_node, learned_node


def nodes_for_learning_three_hyperparams(learned_nodes):
    learned_node = InvGammaNode(0.1, name='HyperVar-Shape', cand_var=var_candsd, shape=1, scale=1)

    mean_node = NormalNode(estimated_mean, name='Mean', cand_var=mean_candsd, mean=learned_nodes[0], var=learned_nodes[1])
    var_node = InvGammaNode(estimated_var, name='Variance', cand_var=var_candsd, shape=learned_node, scale=varprior_scale)
    for datum in data:
        NormalNode(datum, observed=True, mean=mean_node, var=var_node)

    return mean_node, var_node, learned_node


def nodes_for_learning_four_hyperparams(learned_nodes):
    learned_node = InvGammaNode(0.1, name='HyperVar-Scale', cand_var=var_candsd, shape=1, scale=1)

    mean_node = NormalNode(estimated_mean, name='Mean', cand_var=mean_candsd, mean=learned_nodes[0], var=learned_nodes[1])
    var_node = InvGammaNode(estimated_var, name='Variance', cand_var=var_candsd, shape=learned_nodes[2], scale=learned_node)
    for datum in data:
        NormalNode(datum, observed=True, mean=mean_node, var=var_node)

    return mean_node, var_node, learned_node


# Perform simulations and plot results

node_plots = {}
learned_nodes = []
for step in range(5):
    mean_node, var_node, learned_node = nodes_for_learning_hyperparam_for_step(step, learned_nodes)
    _log.info("Learning hyperparam {} (step {})...".format(learned_node, step))
    for datum in data:
        NormalNode(datum, observed=True, mean=mean_node, var=var_node)

    network = Network([mean_node] + mean_node.parents + [var_node] + var_node.parents)
    samples = network.collect_samples(burn, nsamples)
    if learned_node is None:
        title = "Mean/Variance with hard-coded hyper-parameters"
    else:
        learned_nodes.append(learned_node)
        title = "Mean/Variance after learning nodes {}".format(", ".join([a_node.display_name for a_node in learned_nodes]))

    mean_var_plot = evilplot.Plot(title=title, xlabel="Mean", ylabel="Variance")

    mean_var_points = list(zip(samples.of_node(mean_node), samples.of_node(var_node)))
    plot_title = "Learning {} hyper-parameter".format(step)
    plot = evilplot.Points(mean_var_points)
    mean_var_plot.append(plot)
    mean_var_plot.show()

    results = {}
    for node in [mean_node, var_node] + learned_nodes:

        params = {
            'mean': numpy.mean(samples.of_node(node)),
            'var': numpy.var(samples.of_node(node)),
            'mode': stats.mode(samples.of_node(node))[0][0]  # first element of first array is the value
        }

        plot_title = "{}: mode = {:.6f}, mean = {:.6f}, var = {:.6f}".format(node.pdf_name,
                                                                             params['mode'],
                                                                             params['mean'],
                                                                             params['var'])
        plot = samples.density_plot_for_node(node, title=plot_title)
        if not node.display_name in node_plots.keys():
            node_plots[node.display_name] = []
        node_plots[node.display_name].append(plot)


for node_name in list(node_plots.keys()):
    combined_plot = evilplot.Plot(title="{}".format(node_name))
    for plot in node_plots[node_name]:
        combined_plot.append(plot)
    combined_plot.show()
