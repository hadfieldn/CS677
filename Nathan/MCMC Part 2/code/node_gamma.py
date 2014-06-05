from node_invgamma import *

_log = logging.getLogger("node_gamma")


class GammaNode(InvGammaNode):
    def __init__(self, value=1, name=None, shape=1, scale=1, shape_modifier=None, cand_var=1, observed=False):
        super().__init__(value=value, name=name, shape=shape, scale=scale,
                         cand_var=cand_var, observed=observed)

        self.shape_modifier = shape_modifier

    def log_current_conditional_probability(self):

        assert(self.current_value > 0)

        shape = Node.parent_node_value(self.shape)
        scale = Node.parent_node_value(self.scale)

        if not self.shape_modifier is None:
            shape = self.shape_modifier(shape)

        p = stats.gamma.pdf(self.current_value, a=shape, scale=1/scale)
        log_p = (0 if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p

