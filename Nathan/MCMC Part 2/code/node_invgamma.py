from node import Node
import logging
import scipy.stats as stats
import math

_log = logging.getLogger("node_invgamma")


class InvGammaNode(Node):
    def __init__(self, value=1, name=None, shape=1, scale=1, cand_var=1, observed=False):
        super().__init__(value=value, name=name, cand_var=cand_var, observed=observed)
        self.shape = shape
        self.scale = scale

        if shape is None:
            raise ValueError("Parameter 'shape' is required")

        if value <= 0:
            raise ValueError("Parameter 'value' must be greater than 0.")

        self.connect_to_parent_node(shape)
        self.connect_to_parent_node(scale)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def pdf_name(self):
        return "{}({}, {})".format(self.display_name, Node.parent_node_str(self.shape), Node.parent_node_str(self.scale))

    def is_candidate_in_domain(self, cand):
        return cand > 0

    def log_current_conditional_probability(self):

        assert(self.current_value > 0)

        shape = Node.parent_node_value(self.shape)
        scale = Node.parent_node_value(self.scale)

        p = stats.invgamma.pdf(self.current_value, a=shape, scale=scale)
        log_p = (0 if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p

