import logging
from node import *
from pydot import pydot

_log = logging.getLogger("network")


class GraphTraverser(object):

    def traverse(self, nodes, visit_func, skip_parents=False, skip_children=False):
        """
        Invokes function 'visit_func' on each node in the graph, passing
        the current node as an argument.
        """
        _visitedNodes = {}
        for node in nodes:
            self._traverse(node, visit_func, _visitedNodes, skip_parents, skip_children)

    def _traverse(self, start_node, visit_func, _visited_nodes=None, skip_parents=False, skip_children=False):

        if _visited_nodes is None:
            _visited_nodes = {}
        elif start_node in _visited_nodes:
            return

        print(start_node.name)

        _visited_nodes[start_node] = True

        if not skip_children:
            for node in start_node.children:
                if isinstance(node, Node) and not node in _visited_nodes:
                    self._traverse(node, visit_func, _visited_nodes, skip_parents, skip_children)

        visit_func(start_node)

        if not skip_parents:
            for node in start_node.parents:
                if isinstance(node, Node) and not node in _visited_nodes:
                    self._traverse(node, visit_func, _visited_nodes, skip_parents, skip_children)


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

            shape = None
            if node.is_bottom_marked and node.is_top_marked:
                shape = "diamond"
            elif node.is_bottom_marked:
                shape = "invtriangle"
            elif node.is_top_marked:
                shape = "triangle"
            if shape:
                dot_node.set('shape', shape)

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

    def set_flag_nodes(self, network, evidence):

        def mark_with_child_flag(node):
            if node.is_observed:
                return
            for child in node.children:
                if child.is_observed:
                    return

            flag = FlagNode()
            flag.connect_to_parent_node(node)

        def mark_with_parent_flag(node):
            if node.is_observed:
                return
            for parent in node.parents:
                if parent.is_observed:
                    return

            flag = FlagNode()
            node.connect_to_parent_node(flag)

        graph = GraphTraverser()
        graph.traverse(evidence, mark_with_parent_flag, skip_parents=True)
        graph.traverse(evidence, mark_with_child_flag, skip_children=True)

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
            node.is_observed = False
            node.is_top_marked = False
            node.is_bottom_marked = False
            node.is_visited = False

        GraphTraverser().traverse(network.nodes, clear_nodes)

        for node in query:
            node.is_query = True

        for node in evidence:
            node.is_observed = True

        # Record the original network
        if graph_filename:
            DotGraph(network).to_png(graph_filename + "-orig.png")

        self.set_flag_nodes(network, evidence)

        #Record the graph after pre-process
        if graph_filename:
            DotGraph(network).to_png(graph_filename + "-after_pre_proc.png")

        FROM_CHILD = True
        FROM_PARENT = False
        scheduled_nodes = [(node, FROM_CHILD) for node in query]
        while len(scheduled_nodes):
            (node, is_from_child) = scheduled_nodes.pop()

            node.is_visited = True
            if is_from_child and not node.is_observed:
                if not node.is_top_marked:
                    node.is_top_marked = True
                    for parent_node in node.parents:
                        scheduled_nodes.append((parent_node, FROM_CHILD))
                if not node.is_bottom_marked:
                    node.is_bottom_marked = True
                    for child_node in node.children:
                        scheduled_nodes.append((child_node, FROM_PARENT))
            elif not is_from_child:
                if node.is_observed and not node.is_top_marked:
                    node.is_top_marked = True
                    for parent_node in node.parents:
                        scheduled_nodes.append((parent_node, FROM_CHILD))
                elif not node.is_observed and not node.is_bottom_marked:
                    node.is_bottom_marked = True
                    for child_node in node.children:
                        scheduled_nodes.append((child_node, FROM_PARENT))

        def mark_pruned_nodes(node):

            """if isinstance(node, FlagNode):
                node.children.remove(node)
                node.parents.remove(node)
            """


            if node.is_observed:
                if not node.is_visited:
                    node.is_pruned = True
            else:
                if not node.is_bottom_marked:
                    node.is_pruned = True
                if not node.is_top_marked:
                    node.is_pruned = True

        GraphTraverser().traverse(network.nodes, mark_pruned_nodes)

        # Record the post-pruning network
        if graph_filename:
            DotGraph(network).to_png(graph_filename + "-post.png")

