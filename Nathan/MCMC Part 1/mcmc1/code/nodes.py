import random
import logging

log = logging.getLogger("nodes")


class Node:

    def __repr__(self):
        return self.__str__()

    def __init__(self, name=None, value=None, parents=[], children=[], is_observed=False):
        self.name = name
        self.parents = parents
        self.children = children
        self.current_value = value
        self.is_observed = is_observed

    def __str__(self):
        return self.display_name()

    @property
    def node_type(self):
        return self.__class__.__name__

    @property
    def display_name(self):
        return self.name if self.name is not None else self.node_type()

    def sample(self):
        """
        Set current_value according to probability given values of all other nodes
        Subclasses must implement this method.
        """
        raise NotImplementedError


    def probability_of_current_value_given_other_nodes(self):
        """
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def current_unnormalized_mb_probability(self):
        p = 1.0
        for node in self.children + [self]:
            p *= node.current_conditional_probability()
        return p

    def current_conditional_probability(self):
        """
        Compute the probability of the current value of this node conditional on the current values of its parents
        """
        parent_values = dict((node, node.current_value) for node in self.parents)
        p = self.probability_of_event(parent_values)

        # p is the probability of the current value being true. If the current
        # value is actually false, the probability is 1-p.
        if not self.current_value:
            p = 1 - p

        return p



class BernoulliNode(Node):

    def __init__(self, name, value=True, parents=[], children=[], prob=None):
        super().__init__(name, value, parents, children)
        self.prob = prob

    def __str__(self):
        val = self.display_name + "(" + str(self.prob) + ") = " + str(self.current_value)
        return val

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

        assert len(self.prob) == 2**len(self.parents), \
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

        table_idx = len(self.prob)-1 - table_idx       # reverse the index to make the first item map to the first node
        p = self.prob[table_idx]

        return p

    def probability_of_current_value_given_other_nodes(self):
        """
        Compute the probability of this node given the probability of all the other nodes.
        Only have to calculate probabilities for nodes in the Markov Blanket (parents, children, parents of children),
        by dividing the conditional probability of its current value by
        its marginal probability.
        """
        saved_value = self.current_value
        num = self.current_unnormalized_mb_probability()

        # calculate marginal probability by adding the current value (True/False) with its opposite (False/True)
        self.current_value = not self.current_value
        denom = num + self.current_unnormalized_mb_probability()

        self.current_value = saved_value

        return num/denom

    def sample(self):
        if not self.is_observed:
            p = self.probability_of_current_value_given_other_nodes()

            # If current value is false, then the probability we calculated is the probability
            # of the node being false. We want the probability of the node being true.
            if self.current_value is False:
                p = 1-p

            r = random.random()
            self.current_value = (r < p)
            log.debug("P(" + self.name + ") = " + str(p))


