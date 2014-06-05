from unittest import TestCase
from nodes import *
from network import *
import numpy
import logging


logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
#logging.getLogger().setLevel(logging.DEBUG)

class TestNetwork(TestCase):

    def test_bernoulli(self):
        b = BernoulliNode(value=False, name='B', prob=[0.001])
        e = BernoulliNode(value=False, name='E', prob=[0.002])
        a = BernoulliNode(value=False, name='A', parents=[b, e], prob=[0.95, 0.94, 0.29, 0.001])
        j = BernoulliNode(value=True, name='J', parents=[a], prob=[0.90, 0.05], observed=True)
        m = BernoulliNode(value=True, name='M', parents=[a], prob=[0.70, 0.01], observed=True)
        network = Network(nodes=[b, e, a, j, m])

        samples = network.collect_samples(burn=0, n=10000)
        log.info("Totals: " + str(samples.totals()))
        self.assertAlmostEqual(0.284, samples.p({b: True}, {j: True, m: True}), delta=0.05)       # Russell & Norvig
        self.assertAlmostEqual(0.716, samples.p({b: False}, {j: True, m: True}), delta=0.05)      # Russell & Norvig
        self.assertAlmostEqual(0.75, samples.p({a: True}, {j: True, m: True}), delta=0.05)        # Seppi
        self.assertAlmostEqual(0.17, samples.p({e: True}, {j: True, m: True}), delta=0.05)        # Seppi

    def test_normal_with_plot(self):

        n = NormalNode(0, "N", mean=0, var=6)
        network = Network(nodes=[n])

        burn = 3000
        num_samples = burn + 10000
        samples = network.collect_samples(burn=burn, n=num_samples, generator=network.metropolis_sample_generator())

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
        samples = network.collect_samples(burn=3000, n=13000, generator=network.metropolis_sample_generator())
        self.assertAlmostEqual(n.mean, numpy.mean(samples.of_node(n)), delta=0.5)
        self.assertAlmostEqual(n.var, numpy.var(samples.of_node(n)), delta=0.5)

        n = NormalNode(10, "N", mean=10, var=2)
        network = Network(nodes=[n])
        samples = network.collect_samples(burn=3000, n=13000, generator=network.metropolis_sample_generator())
        self.assertAlmostEqual(n.mean, numpy.mean(samples.of_node(n)), delta=0.5)
        self.assertAlmostEqual(n.var, numpy.var(samples.of_node(n)), delta=0.5)

        # If value is observed, make sure we get it for the mean, with zero variance
        n.is_observed = True
        n.current_value = 8.3
        samples = network.collect_samples(burn=0, n=100, generator=network.metropolis_sample_generator())
        self.assertAlmostEqual(8.3, numpy.mean(samples.of_node(n)), places=10)
        self.assertAlmostEqual(0, numpy.var(samples.of_node(n)), places=10)

    def test_normal_with_two_nodes(self):

        m = NormalNode(0, "mu", mean=2, var=3)
        n = NormalNode(-4.0, "N", mean=m, var=6, observed=False)
        network = Network(nodes=[m, n])

        burn = 3000
        num_samples = burn + 10000
        samples = network.collect_samples(burn=burn, n=num_samples, generator=network.metropolis_sample_generator())

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