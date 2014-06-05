from unittest import TestCase
from node_normal import *
from node_beta import *
from node_gamma import *
from node_poisson import *
from node_bernoulli import *
from network import *
import numpy
import logging

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')


class TestNetwork(TestCase):

    def test_normal_with_plot(self):

        n = NormalNode(0, "N", mean=0, var=6)
        network = Network(nodes=[n])

        burn = 3000
        num_samples = burn + 10000
        samples = network.collect_samples(burn=burn, n=num_samples)

        mean = numpy.mean(samples.of_node(n))
        var = numpy.var(samples.of_node(n))
        title = "N({}, {}): mean = {}, var = {} (burn={})".format(n.mean, n.var, mean, var, burn)
        samples.plot_node(n, title=title)
        samples.plot_histogram_for_node(n, title=title)

        self.assertAlmostEqual(n.mean, mean, delta=0.5)
        self.assertAlmostEqual(n.var, var, delta=0.5)

    def test_normal(self):
        n = NormalNode(0, "N", mean=0, var=6)
        network = Network(nodes=[n])
        samples = network.collect_samples(burn=3000, n=13000)
        self.assertAlmostEqual(n.mean, numpy.mean(samples.of_node(n)), delta=0.5)
        self.assertAlmostEqual(n.var, numpy.var(samples.of_node(n)), delta=0.5)

        n = NormalNode(10, "N", mean=10, var=2)
        network = Network(nodes=[n])
        samples = network.collect_samples(burn=3000, n=13000)
        self.assertAlmostEqual(n.mean, numpy.mean(samples.of_node(n)), delta=0.5)
        self.assertAlmostEqual(n.var, numpy.var(samples.of_node(n)), delta=0.5)

        # If value is observed, make sure we get it for the mean, with zero variance
        n.is_observed = True
        n.current_value = 8.3
        samples = network.collect_samples(burn=0, n=100)
        self.assertAlmostEqual(8.3, numpy.mean(samples.of_node(n)), places=10)
        self.assertAlmostEqual(0, numpy.var(samples.of_node(n)), places=10)

    def test_normal_with_two_nodes(self):

        m = NormalNode(0, "mu", mean=2, var=3)
        n = NormalNode(-4.0, "N", mean=m, var=6, observed=False)
        network = Network(nodes=[m, n])

        burn = 3000
        num_samples = burn + 10000
        samples = network.collect_samples(burn=burn, n=num_samples)

        results = {}
        for node in [m, n]:
            params = {
                'mean': numpy.mean(samples.of_node(node)),
                'var': numpy.var(samples.of_node(node))
            }
            results[node] = params

            title = "{}: mean = {}, var = {} (burn={}, n={})".format(node.pdf_name, params['mean'], params['var'],
                                                                     burn, num_samples - burn)
            samples.plot_node(node, title=title)
            if params['var'] > 0:         # histogram fails if all values are the same
                samples.plot_histogram_for_node(node, title=title)

        self.assertAlmostEqual(m.mean, results[m]['mean'], delta=0.5)
        self.assertAlmostEqual(m.var, results[m]['var'], delta=0.5)

        self.assertAlmostEqual(n.mean, results[n]['mean'], delta=0.5)
        self.assertAlmostEqual(n.var, results[n]['var'], delta=0.5)

    def test_invgamma_with_plot(self):

        # to compute mean and variance, shape must be greater than 2 (for mean only, greater than 1)
        n = InvGammaNode(1, "IGamma", shape=2.2, scale=1/2.5, cand_var=0.2)
        network = Network(nodes=[n])

        burn = 0
        num_samples = burn + 50000
        samples = network.collect_samples(burn=burn, n=num_samples)

        mean = numpy.mean(samples.of_node(n))
        var = numpy.var(samples.of_node(n))
        title = "IGamma({}, {}): mean = {}, var = {} (burn={}, n={})".format(n.shape, n.scale, mean, var,
                                                                             burn, num_samples - burn)
        samples.plot_node(n, title=title)
        samples.plot_histogram_for_node(n, title=title)

        if n.shape > 1:
            self.assertAlmostEqual(n.scale/(n.shape-1), mean, delta=0.5)

        if n.shape > 2:
            self.assertAlmostEqual(n.scale**2/((n.shape-1)**2 * (n.shape-2)), var, delta=0.5)

    def test_gamma_with_plot(self):

        # to compute mean and variance, shape must be greater than 2 (for mean only, greater than 1)
        n = GammaNode(10, "Gamma", shape=9, scale=1/0.5, cand_var=2)
        network = Network(nodes=[n])

        burn = 10000
        num_samples = burn + 10000
        samples = network.collect_samples(burn=burn, n=num_samples)

        mean = numpy.mean(samples.of_node(n))
        var = numpy.var(samples.of_node(n))
        title = "Gamma({}, {}): mean = {}, var = {} (burn={}, n={})".format(n.shape, n.scale, mean, var,
                                                                             burn, num_samples - burn)
        samples.plot_node(n, title=title)
        samples.plot_histogram_for_node(n, title=title)

        self.assertAlmostEqual(n.shape * (1/n.scale), mean, delta=0.5)
        self.assertAlmostEqual(n.shape * (1/n.scale)**2, var, delta=0.5)

    def test_beta_with_plot(self):

        n = BetaNode(0.1, "Beta", alpha=2, beta=2, cand_var=0.1)
        network = Network(nodes=[n])

        burn = 10000
        num_samples = burn + 10000
        samples = network.collect_samples(burn=burn, n=num_samples)

        mean = numpy.mean(samples.of_node(n))
        var = numpy.var(samples.of_node(n))
        title = "Beta({}, {}): mean = {}, var = {} (burn={}, n={})".format(n.alpha, n.beta, mean, var,
                                                                            burn, num_samples - burn)
        samples.plot_node(n, title=title)
        samples.plot_histogram_for_node(n, title=title)

        self.assertAlmostEqual(n.alpha/(n.alpha+n.beta), mean, delta=0.5)
        self.assertAlmostEqual(n.alpha * n.beta / ((n.alpha+n.beta)**2*(n.alpha+n.beta+1)), var, delta=0.5)

    def test_poisson_with_plot(self):

        n = PoissonNode(10, "Poisson", rate=4, cand_var=5)
        network = Network(nodes=[n])

        burn = 10000
        num_samples = burn + 100000
        samples = network.collect_samples(burn=burn, n=num_samples)

        mean = numpy.mean(samples.of_node(n))
        var = numpy.var(samples.of_node(n))
        title = "Poisson({}): mean = {}, var = {} (burn={}, n={})".format(n.rate, mean, var,
                                                                            burn, num_samples - burn)
        samples.plot_node(n, title=title)
        samples.plot_histogram_for_node(n, title=title)

        self.assertAlmostEqual(n.rate, mean, delta=0.5)
        self.assertAlmostEqual(n.rate, var, delta=0.5)

    def test_bernoulli_with_plot(self):

        n = BernoulliNode(0.5, "Poisson", p=0.5)
        network = Network(nodes=[n])

        burn = 0
        num_samples = burn + 100000
        samples = network.collect_samples(burn=burn, n=num_samples)

        mean = numpy.mean(samples.of_node(n))
        var = numpy.var(samples.of_node(n))
        title = "Bernoulli({}): mean = {}, var = {} (burn={}, n={})".format(n.p, mean, var,
                                                                          burn, num_samples - burn)
        samples.plot_node(n, title=title)
        samples.plot_histogram_for_node(n, title=title)

        self.assertAlmostEqual(n.p, mean, delta=0.01)
        self.assertAlmostEqual(n.p * (1-n.p), var, delta=0.01)
