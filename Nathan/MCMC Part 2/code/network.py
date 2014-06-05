import logging
import evilplot

log = logging.getLogger("network")


class Network(object):

    def __init__(self, nodes=None):
        self.nodes = [] if nodes is None else nodes

    def __str__(self):
        pass

    def metropolis_sample_generator(self):
        """Create samples from the given nodes using the Metrpolis algorithm."""

        while True:
            for test_node in self.nodes:
                test_node.sample_with_metropolis()

                network_state = []
                for node in self.nodes:
                    network_state.append(node.current_value)
                yield network_state

    def collect_samples(self, burn, n, generator=None):
        """Run burn iterations, then collect n samples"""

        mcmc = generator
        if mcmc is None:
            mcmc = self.metropolis_sample_generator()

        progress_step = (burn + n) / 10
        cur_sample = 0

        log.info("Burning...")
        for i in range(burn):
            next(mcmc)
            cur_sample += 1
            if cur_sample % progress_step == 0:
                log.warning("{:.0%}... ".format(cur_sample/(burn+n)))


        log.info("Sampling...")
        samples = []
        for i in range(n):
            sample = next(mcmc)
            log.debug("Sample: " + str(sample))
            samples.append(next(mcmc))
            cur_sample += 1
            if cur_sample % progress_step == 0:
                log.warning("{:.0%}... ".format(cur_sample/(burn+n)))

        return SamplesProcessor(self.nodes, samples)


class SamplesProcessor(object):

    def __init__(self, nodes, samples):
        if not type(nodes) is list:
            raise AssertionError("'nodes' argument is not a list (type = " + type(nodes).__name__ + ")")
        self.nodes = nodes
        self.samples = samples

    def __str__(self):
        samples_str = ", ".join([node.name for node in self.nodes]) + "\n"
        samples_str += "\n".join([", ".join(map(str, sample)) for sample in self.samples])
        return samples_str

    def of_node(self, node):
        """Returns samples for the given node"""

        samples = [sample[self.nodes.index(node)] for sample in self.samples]
        return samples

    def plot_node(self, node, title=None):
        if title is None:
            title = u"Samples of {0:s}".format(node.display_name)
        p = evilplot.Plot(title=title)

        points = evilplot.Points(list(enumerate(self.of_node(node))))
        points.style = 'lines'
        points.linewidth = 1
        p.append(points)
        p.show()

    def plot_histogram_for_node(self, node, title=None, prior_pdf=None):
        if title is None:
            title = u"Histogram of samples of {0:s}".format(node.display_name)
        p = evilplot.Plot(title=title)

        if not prior_pdf is None:
            priord = evilplot.Function(prior_pdf)
            priord.title = "Prior Dist"
            p.append(priord)

        hist = evilplot.Histogram(self.of_node(node), 50, normalize=True)
        hist.title = node.display_name
        p.append(hist)
        p.show()