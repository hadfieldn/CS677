__author__ = 'nathan'

from node_normal import *
from node_beta import *
from node_gamma import *
from node_poisson import *
from network import *
import numpy

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')


burn = 0
num_samples = burn + 100000


for d_observed in [False,True]:
    a = BetaNode(0.4, 'A', alpha=2, beta=2)
    b = GammaNode(4, 'B', shape=3, scale=1/2)
    c = NormalNode(0, 'C', mean=b, var=a)
    d = PoissonNode(5, 'D', rate=b, observed=d_observed)

    network = Network([a, b, c, d])
    samples = network.collect_samples(burn=burn, n=num_samples)

    for node in [a, b, c, d]:
        mean = numpy.mean(samples.of_node(node))
        var = numpy.var(samples.of_node(node))
        title = "{} [D observed = {}]: mean = {:.4f}, var = {:.4f} (burn={}, n={})".format(node.pdf_name, d_observed,
                                                                                         mean, var, burn, num_samples-burn)
        samples.plot_node(node, title=title)
        samples.plot_histogram_for_node(node, title=title)

