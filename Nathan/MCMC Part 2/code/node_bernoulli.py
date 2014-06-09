from node import Node
import logging
import math
import random
import scipy.stats as stats

_log = logging.getLogger("node_bernoulli")


class BernoulliNode(Node):
    def __init__(self, value=1, name=None, p=0.5, observed=False):
        super().__init__(value=value, name=name, cand_var=1, observed=observed)
        if not isinstance(p, list):
            p = [p]

        self.p = p

        if value < 0 or value > 1:
            raise ValueError("Parameter 'value' must be between 0 and 1.")

        for parent in self.p:
            self.connect_to_parent_node(parent)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def pdf_name(self):
        return "{}({})".format(self.display_name, Node.parent_node_str(self.p))

    def select_candidate(self):
        # p = Node.parent_node_value(self.p)
        sample = 1 if random.random() <= 0.5 else 0
        return sample

    def log_current_conditional_probability(self):
        """
        For Bernoulli/Binomial, sample directly instead of trying to use Metropolis.
        """
        param_p = Node.parent_node_value(self.p)

        p = stats.bernoulli.pmf(self.current_value, param_p)
        log_p = (Node.IMPOSSIBLE if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p

