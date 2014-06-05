from node import Node
import logging
import scipy.stats as stats
import math

_log = logging.getLogger("node_beta")


class BetaNode(Node):
    def __init__(self, value=1, name=None, alpha=1, beta=1, cand_var=1, observed=False):
        super().__init__(value=value, name=name, cand_var=cand_var, observed=observed)
        self.alpha = alpha
        self.beta = beta

        if value < 0 or value > 1:
            raise ValueError("Parameter 'value' must be greater than 0 and less than 1.")

        self.connect_to_parent_node(alpha)
        self.connect_to_parent_node(beta)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def pdf_name(self):
        return "{}({}, {})".format(self.display_name, Node.parent_node_str(self.alpha), Node.parent_node_str(self.beta))

    def is_candidate_in_domain(self, cand):
        return 0 <= cand <= 1

    def log_current_conditional_probability(self):

        assert(self.current_value > 0)

        alpha = Node.parent_node_value(self.alpha)
        beta = Node.parent_node_value(self.beta)

        p = stats.beta.pdf(self.current_value, a=alpha, b=beta)
        log_p = (0 if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p


