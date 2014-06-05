from node_normal import *
from node_beta import *
from node_gamma import *
from node_poisson import *
from node_bernoulli import *
from network import *
import numpy

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')


burn = 0
num_samples = burn + 100000

for g_observed in [False, True]:
    a = NormalNode(20, 'A', mean=20, var=1)
    e = BetaNode(0.5, 'E', alpha=1, beta=1)
    b = GammaNode(0.2, 'B', shape=a, shape_modifier=lambda x: x ** math.pi, scale=1/7)
    d = BetaNode(0.5, 'D', alpha=a, beta=e)
    c = BernoulliNode(0, 'C', p=d)
    f = PoissonNode(4, 'F', rate=d)
    g = NormalNode(5, 'G', mean=e, var=f, observed=g_observed)

    network = Network([a, e, b, d, c, f, g])
    samples = network.collect_samples(burn=burn, n=num_samples)

    for node in network.nodes:
        mean = numpy.mean(samples.of_node(node))
        var = numpy.var(samples.of_node(node))
        title = "{} [G observed={}]: mean = {:.4f}, var = {:.4f} (burn={}, n={})".format(node.pdf_name, g_observed,
                                                                                 mean, var, burn, num_samples-burn)
        samples.plot_node(node, title=title)
        samples.plot_histogram_for_node(node, title=title)

