import random
import logging
import math

_log = logging.getLogger("nodes")


class Node:
    def __repr__(self):
        return self.__str__()

    def __init__(self, value=None, name=None, cand_var=1, observed=False):
        self.name = name
        self.current_value = value
        self.cand_std_dev = math.sqrt(cand_var)     # std_dev of Gaussian distribution used to generate candidates
        self.is_observed = observed

        self._children = []  # subclass init methods should add self to parents' children
        self._log_p_current_value = None  # log of the last sample

    def __str__(self):
        return self.display_name()

    @property
    def pdf_name(self):
        return self.display_name()

    @property
    def node_type(self):
        return self.__class__.__name__

    @property
    def display_name(self):
        return self.name if not self.name is None else self.node_type

    @staticmethod
    def parent_node_str(node):
        return "{:.4f}".format(node) if not isinstance(node, Node) else node.display_name

    @staticmethod
    def parent_node_value(node):
        """
        If node is a list of parent nodes, returns the sum of their values.
        """
        if isinstance(node, Node):
            return node.current_value
        elif isinstance(node, list):
            return sum([Node.parent_node_value(a_node) for a_node in node])
        else:
            return node

    def connect_to_parent_node(self, parent):
        """
        If parent is a list nodes, connects to each of them.
        """
        if isinstance(parent, Node):
            parent._children.append(self)
        elif isinstance(parent, list):
            for parentnode in parent:
                self.connect_to_parent_node(parentnode)

    def current_conditional_probability(self):
        """Provided for testing; use log_current_conditional_probability instead."""
        return math.exp(self.log_current_conditional_probability())

    def log_current_conditional_probability(self):
        """Compute the conditional probability of this node given its parents"""
        raise NotImplementedError

    def current_unnormalized_mb_probability(self):
        """Provided for testing; use log_current_unnormalized_mb_probability instead."""
        return math.exp(self.log_current_unnormalized_mb_probability())

    def log_current_unnormalized_mb_probability(self):
        p = 0.0
        for node in self._children + [self]:
            p += node.log_current_conditional_probability()
        return p

    def probability_of_current_value_given_other_nodes(self):
        return math.exp(self.log_probability_of_current_value_given_other_nodes())

    def log_probability_of_current_value_given_other_nodes(self):
        """
        Needed only for Gibbs sampling. Metropolis sampling only requires
        a probability that is proportional to the actual probability, which
        saves us from having to determine the integral for the marginal
        probability.
        """
        raise NotImplementedError

    def is_candidate_in_domain(self, cand):
        """Overridden by subclasses to reject samples that are outside the domain of the probability function."""
        return True

    def select_candidate(self):
        """Can be overridden by subclasses in order to provide custom distributions. Default is Gaussian."""
        return random.gauss(self.current_value, self.cand_std_dev)

    def sample_with_gibbs(self):
        """
        Samples boolean values.
        """
        if not self.is_observed:
            p = self.probability_of_current_value_given_other_nodes()

            r = random.random()
            self.current_value = (r < p)
            _log.debug("P(" + self.name + ") = " + str(p))

    def sample_with_metropolis(self):
        """Sample this node using Metropolis."""

        _log.debug("Sampling {}...".format(self))

        if not self.is_observed:

            # Metropolis:
            # 1 - Use the candidate distribution to select a candidate.
            # 2 - Compare the (proportionate) probability of the candidate with the
            # (proportionate) probability of the current value.
            # 3 - If the probability of the candidate is greater, use it.
            # Otherwise, determine whether to use it as a random selection with
            # probability proportionate to the probability of the current value.

            # 1 - Select a candidate. (Since we're not using Metropolis-Hastings,
            # we use a Gaussian normal with variance provided by parameter 'cand_var'.)

            cand = self.select_candidate()
            _log.debug("last: {}, cand: {}".format(self.current_value, cand))

            # If the candidate falls outside the domain of the probability function,
            # we can skip it immediately.
            if self.is_candidate_in_domain(cand):
                # 2 - Compare the probability of the candidate with that of the current value

                # log_p_cand = candidate probability
                saved_value = self.current_value
                self.current_value = cand
                log_p_cand = self.log_current_unnormalized_mb_probability()
                self.current_value = saved_value

                # log_p_current_value = current probability
                if self._log_p_current_value is None:
                    self._log_p_current_value = self.log_current_unnormalized_mb_probability()


                log_r = log_p_cand - self._log_p_current_value
                log_u = math.log(random.random())

                _log.debug("log_r = {}, log_u = {}".format(log_r, log_u))

                # 3 - Use candidate with probability proportionate to the ratio of
                # its likelihood over the likelihood of the current value.

                if log_u < log_r:
                    self.current_value = cand
                    self._log_p_current_value = log_p_cand




