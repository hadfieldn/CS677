import logging
from node import *
from pydot import pydot

_log = logging.getLogger("network")


class GraphTraverser(object):

    def traverse(self, nodes, visit_func):
        """
        Invokes function 'visit_func' on each node in the graph, passing
        the current node as an argument.
        """
        _visitedNodes = {}
        for node in nodes:
            self._traverse(node, visit_func, _visitedNodes)

    def _traverse(self, start_node, visit_func, _visited_nodes=None):

        if _visited_nodes is None:
            _visited_nodes = {}
        elif start_node in _visited_nodes:
            return

        _visited_nodes[start_node] = True

        for node in start_node.children:
            if isinstance(node, Node) and not node in _visited_nodes:
                self._traverse(node, visit_func, _visited_nodes)

        visit_func(start_node)

        for node in start_node.parents:
            if isinstance(node, Node) and not node in _visited_nodes:
                self._traverse(node, visit_func, _visited_nodes)


class DotGraph(object):
    """
    Uses pydot to generate a dot-language representation of the network.
    Can be written as a PNG or SVG image, or a raw text file.
    """

    def __init__(self, network):
        self.network = network

    def __str__(self):
        pass

    @staticmethod
    def escape_string(string):
        return string.replace("\"", "\\\"")

    def graph(self):
        g = pydot.Dot(graph_type='digraph', graph_name=id(self.network))

        def add_node_to_dot_graph(node, g=g):

            dot_node = pydot.Node(id(node))

            node_name = node.display_name
            if node.is_observed:
                node_name += " = {}".format(node.current_value)
            dot_node.set('label', node_name)


            if node.is_pruned:
                dot_node.set('color', 'red')
                dot_node.set('fontcolor', 'red')
                if node.is_observed:
                    dot_node.set('style', 'filled, dashed')
                    dot_node.set('fillcolor', "#ff000020")
                elif node.is_query:
                    dot_node.set('style', 'bold, dashed')
                else:
                    dot_node.set('style', 'dashed')

            else:
                if node.is_observed:
                    dot_node.set('style', 'filled')
                elif node.is_query:
                    dot_node.set('style', 'bold')

            g.add_node(dot_node)

            for parent_node in node.parents:
                edge = pydot.Edge(id(parent_node), id(node))
                if node.is_pruned or parent_node.is_pruned:
                    edge.set('color', 'red')
                    edge.set('style', 'dashed')
                g.add_edge(edge)

        gt = GraphTraverser()
        gt.traverse(self.network.nodes, add_node_to_dot_graph)

        if self.network.name:
            g.set('labelloc', 't')
            g.set('label', self.network.name)

        return g

    def to_string(self):
        return self.graph().to_string()

    def to_file(self, path, format="raw"):
        self.graph().write(path, format=format)

    def to_png(self, path):
        self.to_file(path, format="png")

    def to_svg(self, path):
        self.to_file(path, format="svg")


class FlagNode(Node):
    """
    Node to represent whether a node's ancestors or descendents include an observed node.
    """
    pass


class Pruner(object):

    def __init__(self):
        pass

    def __str__(self):
        pass

    def set_flag_nodes(self, network, query, evidence):
        pass

    def prune(self, network, query=None, evidence=None, graph_filename=None):
        """
        Given a network, a list of query nodes, and a list of evidence nodes,
        prune the nodes that are unnecessary to sample. (Rather than actually
        removing the nodes, we set node.is_pruned = True so that we can still
        include them in the Dot graph representation (identifying them by coloring
        them red). (Network.metropolis_sample_generator() will skip nodes for
        which is_pruned = True.)

        :param network:
        :param query:
        :param evidence:
        :param graph_filename:
        :return:
        """

        if not query:
            query = []

        if not evidence:
            evidence = []

        all_nodes = []

        # clear any flags from previous prunings
        def clear_nodes(node):
            node.is_query = False
            node.is_pruned = False
            node.is_observe = False
            node.is_top_visited = False
            node.is_bottom_visited = False

        GraphTraverser().traverse(network.nodes, clear_nodes)

        for node in query:
            node.is_query = True

        for node in evidence:
            node.is_observed = True

        # Record the pre-pruning network
        if graph_filename:
            DotGraph(network).to_png(graph_filename + "-pre.png")

        # TODO: Pruning algorithm goes here

