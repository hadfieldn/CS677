from node import Node
import logging
import math
import random

_log = logging.getLogger("node_bernoulli")


class BernoulliNode(Node):
    def __init__(self, value=1, name=None, p=0.5, observed=False):
        super().__init__(value=value, name=name, cand_var=1, observed=observed)
        self.p = p

        if value < 0 or value > 1:
            raise ValueError("Parameter 'value' must be between 0 and 1.")

        self.connect_to_parent_node(p)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def pdf_name(self):
        return "{}({})".format(self.display_name, Node.parent_node_str(self.p))


    def log_current_conditional_probability(self):
        """
        For Bernoulli/Binomial, sample directly instead of trying to use Metropolis.
        """

        p = Node.parent_node_value(self.p)
        sample = 1 if random.random() <= p else 0
        log_sample = (0 if sample == 0 else math.log(sample))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, sample))
        return log_sample

