from node_normal import *
from node_beta import *
from node_gamma import *
from node_poisson import *
from node_bernoulli import *
from network import *
import numpy

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')


burn = 10000
num_samples = burn + 100000

"""
Node B needs lots of iterations to become stable. Try 100,000.
"""
for g_observed in [False]: #, True]:

    # Plot[PDF[NormalDistribution[20, 1], x], {x, 17, 23}]
    a = NormalNode(20, 'A', mean=20, var=1)

    # Plot[PDF[BetaDistribution[1, 1], x], {x, 0, 1}, PlotRange -> All]
    # Flat (p=1.0) -- so this node moves all over the place!
    e = BetaNode(0.5, 'E', alpha=1, beta=1)

    # Manipulate[Plot[PDF[GammaDistribution[a^Pi, 1/7], x], {x, 0, 5000}, PlotRange -> 0.1], {a, 16, 24}]
    # This node converges fairly tightly with nearly equal probability on on some value between about
    # 1,000 and 3,000; most likely, somewhere around 1,800 (when a == 20, its mean).
    b = GammaNode(1600, 'B', shape=a, shape_modifier=lambda x: x ** math.pi, scale=1/7)

    # Manipulate[Plot[PDF[BetaDistribution[alpha, beta], x], {x, 0.0001, 1},PlotRange -> All], {alpha, 16, 24}, {beta, 0.0001, 1}]
    # Node a (alpha) doesn't influence this distribution much; it is VERY likely to return values
    # close to 1, with slightly higher probability of getting lower numbers when node e (beta) is closer to 0 than 1.
    d = BetaNode(0.8, 'D', alpha=a, beta=e)

    # Manipulate[DiscretePlot[PDF[BernoulliDistribution[p], x], {x, 0, 1}, PlotRange -> All], {p, 0, 1}]
    # Since d is likely to be close to 1, c will almost always be True
    c = BernoulliNode(0, 'C', p=d)

    # Manipulate[DiscretePlot[PDF[PoissonDistribution[rate], x], {x, 0, 10},PlotRange -> All, PlotMarkers -> Automatic], {rate, 0.001, 1}]
    # When rate is close to 1, most likely values are 0 and 1, with some probability of 2 and slight probability of 3, and very slight
    # probability of 4. For lower values of rate, skew is even closer toward 0.
    f = PoissonNode(0.1, 'F', rate=d)

    # Manipulate[Plot[PDF[NormalDistribution[mean, Sqrt[var]], x], {x, -4, 5}, PlotRange -> {0, 1}], {mean, 0, 1}, {var, 0.0001, 4}]
    # Mean is close to 1, variance is likely to be 0 or 1, somewhat likely to be 2 or 3 -- so values are likely to be somewhere
    # between -5 and 5, much closer to 1 when f is 1. (When f is 0, Normal is undefined.)
    g = NormalNode(5, 'G', mean=e, var=f, observed=g_observed)

    network = Network([a, e, b, d, c, f, g])
    samples = network.collect_samples(burn=burn, n=num_samples)

    for node in reversed(network.nodes):
        mean = numpy.mean(samples.of_node(node))
        var = numpy.var(samples.of_node(node))
        title = "{} [G observed={}]: mean = {:.4f}, var = {:.4f} (burn={}, n={})".format(node.pdf_name, g_observed,
                                                                                 mean, var, burn, num_samples-burn)
        samples.plot_node(node, title=title)
        if var > 0:
            samples.plot_histogram_for_node(node, title=title)

