from node import Node
import logging
import math
import random
import scipy.stats as stats

_log = logging.getLogger("node_bernoulli")


class BernoulliNode(Node):
    def __init__(self, value=1, name=None, parents=None, p=None, observed=False):
        """
        Bernoulli distribution. 0 = False, 1 = True.
        :param:parents: List of parent Bernoulli nodes (optional)
        :param:p: List of probability entries for parent values (must have 2**len(parents) entries)
        """
        super().__init__(value=value, name=name, cand_var=1, observed=observed)

        if not isinstance(value, bool) and value < 0 or value > 1:
            raise ValueError("Initial value 'value' must a boolean or a float between 0 and 1.")

        if not isinstance(p, list):
            p = [p]

        if parents is None:
            parents = []

        if len(p) != 2**len(parents):
            raise ValueError("List 'p' must have entries for 2**len(parents) = {}".format(2**len(parents)))

        self.p = p
        self.bernoulli_parents = parents

        for node in [self.p] + [self.bernoulli_parents]:
            self.connect_to_parent_node(node)

    def __str__(self):
        return "{} = {}".format(self.pdf_name, self.current_value)

    @property
    def pdf_name(self):
        return "{}({})".format(self.display_name, Node.parent_node_str(self.p))

    def select_candidate(self):
        # p = Node.parent_node_value(self.p)
        # If node has parents, they should be Bernoulli nodes
        sample = 1 if random.random() <= 0.5 else 0
        return sample

    def probability_of_event(self, event):
        """
            Calculate probability of node/values given in event dict.
            Nodes must be contained within 'parents' dictionary.

            Probability table has 2^n rows. E.g., if parents are A, B:

            A=true, B=true = prob[0]
            A=true, B=false = prob[1]
            A=false, B=true = prob[2]
            A=false, B=false = prob[3]
        """

        assert len(self.p) == 2**len(self.bernoulli_parents), \
            "Prob table for Bernoulli node '" + self.display_name + "' does not have enough entries for its " \
            + str(len(self.bernoulli_parents)) + " parents."

        table_idx = 0
        for bernoulli_parent in self.bernoulli_parents:
            table_idx *= 2
            parent_event = event[bernoulli_parent]       # event is dictionary of bernoulli parent nodes and their current values
            if parent_event != 0.0 and parent_event != 1.0:
                _log.warn(("Current value '{}' of parent '{}' of Bernoulli node '{}''"
                           " should be either 1 (True) or 0 (False). Treating it"
                           " as 0 instead.")
                          .format(parent_event, bernoulli_parent.display_name, self.display_name))
            elif parent_event == 1.0:
                table_idx += 1

        assert table_idx < len(self.p)
        table_idx = len(self.p)-1 - table_idx       # reverse the index to make the first item map to the first node
        prob = Node.parent_node_value(self.p[table_idx])

        return prob

    def log_current_conditional_probability(self):
        """
        For Bernoulli/Binomial, sample directly instead of trying to use Metropolis.
        """
        parent_values = dict((node, Node.parent_node_value(node)) for node in self.bernoulli_parents)
        p = self.probability_of_event(parent_values)
        if( self.current_value == 0 ):
            p = 1-p
        # p = stats.bernoulli.pmf(self.current_value, param_p)
        log_p = (Node.IMPOSSIBLE if p == 0 else math.log(p))

        _log.debug("p({}={}) = {}".format(self.display_name, self.current_value, p))
        return log_p

