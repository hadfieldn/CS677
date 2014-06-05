import logging
import evilplot

log = logging.getLogger("network")


class Network(object):

    def __init__(self, nodes=None):
        self.nodes = [] if nodes is None else nodes

    def __str__(self):
        pass

    def gibbs_sample_generator(self):
        """Create samples from the given nodes using the Gibbs algorithm."""

        while True:
            for test_node in self.nodes:
                test_node.sample_with_gibbs()

                network_state = []
                for node in self.nodes:
                    network_state.append(node.current_value)
                yield network_state

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


    def is_sample_match(self, sample, event):
        """
        Returns true if the value mapped to each node in 'event' matches the corresponding value of 'sample'.
        NOTE: Should only be called for samples of boolean values.
        :param sample:
        :param event:
        :return:
        """

        is_match = False
        for idx, node in enumerate(self.nodes):
            if node in event:
                if sample[idx] != event[node]:
                    break
        else:
            is_match = True

        return is_match

    def totals(self, start=None, end=None):

        if start is None:
            start = 0
        if end is None:
            end = len(self.samples)

        num_nodes = len(self.nodes)
        totals = [0] * num_nodes
        for i in range(start, end):
            sample = self.samples[i]
            for idx in range(num_nodes):
                if sample[idx]:
                    totals[idx] += 1

        return totals

    def p(self, outcomes, givens, start=None, end=None):
        """
        :param outcome: dictionary of nodes and values
        :param given: dictionary of nodes and values
        :return: probability (float in range[0..1])
        """

        if start is None:
            start = 0
        if end is None:
            end = len(self.samples)

        matching_givens_count = 0
        matching_outcomes_count = 0

        outcomes_and_givens = {}
        for d in [outcomes, givens]:
            outcomes_and_givens.update(d)

        for i in range(start, end):
            sample = self.samples[i]
            if self.is_sample_match(sample, givens):
                matching_givens_count += 1
            if self.is_sample_match(sample, outcomes_and_givens):
                matching_outcomes_count += 1

        p = 0 if matching_givens_count == 0 else matching_outcomes_count / matching_givens_count

        return p

    def plot_mixing(self, name, outcomes, givens):

        prob_samples = [self.p(outcomes, givens, 0, i) for i in range(len(self.samples))]

        p = evilplot.Plot(title=u"{0:s}".format(name))
        points = evilplot.Points(list(enumerate(prob_samples)))
        points.style = 'lines'
        points.linewidth = 1
        p.append(points)
        #p.write_gpi('plots/mix-%s.gpi' % name)
        p.show()

    def plot_histogram(self, name, outcomes, givens):

        prob_samples = [self.p(outcomes, givens, 0, i) for i in range(len(self.samples))]

        p = evilplot.Plot(title=u"{0:s}".format(name))

        hist = evilplot.Histogram(prob_samples, 100, normalize=True)
        hist.title = 'Posterior Dist'
        p.append(hist)

        p.show()

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