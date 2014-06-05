import random
import logging
import scipy
import math
import numpy.random as rnd
import scipy.stats as stats

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

    def log_probability_of_current_value_given_other_nodes(self):
        """
        Needed only for Gibbs sampling. Metropolis sampling only requires
        a probability that is proportional to the actual probability, which
        saves us from having to determine the integral for the marginal
        probability.
        """
        raise NotImplementedError

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

            cand = random.gauss(self.current_value, self.cand_std_dev)
            _log.debug("last: {}, cand: {}".format(self.current_value, cand))

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


class BernoulliNode(Node):
    def __init__(self, value=True, name=None, parents=[], prob=None, observed=False):
        """
        :param name: Name of the node
        :param value: Initial value of the node
        :param parents: Other nodes this node is conditional upon
        :param prob: list of probabilities for each combination of parent values
        :param observed: (Optional) Whether the current node is an observed value.
         If observed, the initial value will be used instead of calculating the probability.
        :return:
        """
        super().__init__(value=value, name=name, observed=observed)
        self.parents = parents
        for parent in self.parents:
            parent.children.append(self)
        self.prob = prob

    def __str__(self):
        val = "{}({}) = {}".format(self.display_name, self.prob, self.current_value)
        return val

    def _probability_of_event(self, event):
        """
            Calculate probability of node/values given in event dict.
            Nodes must be contained within 'parents' dictionary.

            Probability table has 2^n rows. E.g., if parents are A, B:

            A=true, B=true = prob[0]
            A=true, B=false = prob[1]
            A=false, B=true = prob[2]
            A=false, B=false = prob[3]
        """

        assert len(self.prob) == 2 ** len(self.parents), \
            "Prob table for Bernoulli node '" + self.display_name + "' does not have enough entries for its " \
            + str(len(self.parents)) + " parents."

        table_idx = 0
        for parent_node in self.parents:
            table_idx *= 2
            parent_event = event[parent_node]
            if parent_event:
                assert isinstance(parent_event, bool), "Current value '" + str(parent_event) \
                                                       + "' of parent '" + parent_node.display_name \
                                                       + "' of Bernoulli node '" + self.display_name \
                                                       + "' is not a boolean."
                table_idx += 1

        assert table_idx < len(self.prob)

        table_idx = len(self.prob) - 1 - table_idx  # reverse the index to make the first item map to the first node
        p = self.prob[table_idx]

        return p

    def log_probability_of_current_value_given_other_nodes(self):
        """
        Compute the probability of this node given the probability of all the other nodes.
        Only have to calculate probabilities for nodes in the Markov Blanket (parents, children, parents of children),
        by dividing the conditional probability of its current value by
        its marginal probability.
        """
        saved_value = self.current_value
        num = math.exp(self.log_current_unnormalized_mb_probability())

        # calculate marginal probability by adding the current value (True/False) with its opposite (False/True)
        self.current_value = not self.current_value
        denom = num + math.exp(self.log_current_unnormalized_mb_probability())
        self.current_value = saved_value

        p = num / denom

        # If current value is false, then the probability we calculated is the probability
        # of the node being false. We want the probability of the node being true.
        if self.current_value is False:
            p = 1 - p

        return math.log(p)

    def log_current_conditional_probability(self):
        """
        Compute the probability of the current value of this node conditional on the current values of its parents
        """
        parent_values = dict((node, node.current_value) for node in self.parents)
        p = self._probability_of_event(parent_values)

        # p is the probability of the current value being true. If the current
        # value is actually false, the probability is 1-p.
        if self.current_value is False:
            p = 1 - p

        return math.log(p)


"""
▪ Normal
▪ Gamma
▪ Inverse Gamma
▪ Poisson
▪ Beta
▪ Bernoulli or Binomial
"""


class NormalNode(Node):
    def __init__(self, value=0, name=None, mean=0, var=1, cand_var=1, observed=False):
        super().__init__(value=value, name=name, cand_var=cand_var, observed=observed)
        self.mean = mean
        self.var = var

        if isinstance(var, Node):
            var._children.append(self)

        if isinstance(mean, Node):
            mean._children.append(self)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def pdf_name(self):
        mean_name = (str(self.mean) if not isinstance(self.mean, Node) else self.mean.display_name)
        var_name = (str(self.var) if not isinstance(self.var, Node) else self.var.display_name)
        val = "{}({}, {})".format(self.display_name, mean_name, var_name)
        return val

    def log_current_conditional_probability(self):
        """
        Return probability given current values of 'mean' and 'var'.
        (If 'mean' and 'var' are parent nodes, get their current_value.)
        """
        mean = (self.mean if not isinstance(self.mean, Node) else self.mean.current_value)
        var = (self.var if not isinstance(self.var, Node) else self.var.current_value)

        p = stats.norm.pdf(self.current_value, mean, math.sqrt(var))
        log_p = (0 if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p

