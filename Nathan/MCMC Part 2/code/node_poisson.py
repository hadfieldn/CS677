from node import Node
import logging
import scipy.stats as stats
import math
import random

_log = logging.getLogger("node_poisson")


class PoissonNode(Node):
    def __init__(self, value=1, name=None, rate=1, cand_var=1, observed=False):
        super().__init__(value=value, name=name, cand_var=cand_var, observed=observed)
        self.rate = rate

        if value <= 0:
            raise ValueError("Parameter 'value' must be greater than 0.")

        self.connect_to_parent_node(rate)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def parents(self):
        parents = []
        if isinstance(self.rate, Node):
            parents.append(self.rate)
        return parents

    @property
    def pdf_name(self):
        return "{}({})".format(self.display_name, Node.param_str(self.rate))

    def is_candidate_in_domain(self, cand):
        return cand >= 0

    def select_candidate(self):
        """For Poisson, use Metropolis with a candidate distribution that rounds samples from a normal."""
        cand = round(random.gauss(self.current_value, self.cand_std_dev), 0)
        return cand


    def log_current_conditional_probability(self):

        assert(self.current_value >= 0)

        rate = Node.param_value(self.rate)

        p = stats.poisson.pmf(self.current_value, mu=rate)
        log_p = (Node.IMPOSSIBLE if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p


