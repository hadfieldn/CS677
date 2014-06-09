from node import Node
import logging
import scipy.stats as stats
import math

_log = logging.getLogger("node_normal")


class NormalNode(Node):
    def __init__(self, value=0, name=None, mean=0, var=1, cand_var=1, observed=False):
        super().__init__(value=value, name=name, cand_var=cand_var, observed=observed)
        self.mean = mean
        self.var = var

        self.connect_to_parent_node(mean)
        self.connect_to_parent_node(var)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def parents(self):
        parents = []
        if isinstance(self.mean, Node):
            parents.append(self.mean)
        if isinstance(self.var, Node):
            parents.append(self.var)
        return parents

    @property
    def pdf_name(self):
        return "{}({}, {})".format(self.display_name, Node.parent_node_str(self.mean), Node.parent_node_str(self.var))

    def is_candidate_in_domain(self, cand):
        return Node.parent_node_value(self.var) > 0

    def log_current_conditional_probability(self):
        """
        Return probability given current values of 'mean' and 'var'.
        (If 'mean' and 'var' are parent nodes, get their current_value.)
        """

        mean = Node.parent_node_value(self.mean)
        var = Node.parent_node_value(self.var)

        if var == 0:
            _log.debug("Node " + str(self) + ": Cannot compute a normal probability when variance is 0.")
            return Node.IMPOSSIBLE

        p = stats.norm.pdf(self.current_value, mean, math.sqrt(var))
        _log.debug("  p = {}".format(p))
        log_p = (Node.IMPOSSIBLE if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p

