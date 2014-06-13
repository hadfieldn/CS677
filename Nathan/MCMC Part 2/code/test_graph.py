from unittest import TestCase
from network import *
from graph import *
from node_normal import *
from node_bernoulli import *
from node_beta import *
from node_gamma import *
from node_poisson import *
from timeit import default_timer as timer
import logging

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
_log = logging.getLogger("test_graph")

class TestDotGraph(TestCase):

    def setUp(self):
        pass

    def test_to_string(self):

        # simple network
        a = NormalNode(0, 'A')
        b = NormalNode(0, 'B')
        c = NormalNode(0, 'C', mean=a, var=b, observed=True)
        network = Network([a, b, c])
        dot_str = DotGraph(network).to_string()
        print(dot_str)

        # alarm network
        b = BernoulliNode(0, name='B', p=[0.2])
        e = BernoulliNode(0, name='E', p=[0.3])
        a = BernoulliNode(0, name='A', parents=[b, e], p=[0.95, 0.94, 0.29, 0.2])
        j = BernoulliNode(0, name='J', parents=[a], p=[0.90, 0.2])
        m = BernoulliNode(0, name='M', parents=[a], p=[0.70, 0.3])
        network = Network(nodes=[b, e, a, j, m], name="Alarm Network")
        dot_str = DotGraph(network).to_string()
        print(dot_str)

        a = NormalNode(20, 'A', mean=20, var=1)
        e = BetaNode(0.5, 'E', alpha=1, beta=1)
        b = GammaNode(1600, 'B', shape=a, shape_modifier=lambda x: x ** math.pi, scale=1/7)
        d = BetaNode(0.8, 'D', alpha=a, beta=e)
        c = BernoulliNode(0, 'C', p=d)
        f = PoissonNode(0.1, 'F', rate=d)
        g = NormalNode(5, 'G', mean=e, var=f, observed=True)
        f.is_pruned = True
        g.is_pruned = True
        network = Network([a, e, b, d, c, f, g], name="\"Wacky\" Network")
        dot_str = DotGraph(network).to_string()
        print(dot_str)

        DotGraph(network).to_png("test_graph_output/wacky.png")
        DotGraph(network).to_svg("test_graph_output/wacky.svg")
        DotGraph(network).to_file("test_graph_output/wacky.gv")


        #self.fail("Have to check this test manually.")


class TestPruner(TestCase):

    def setUp(self):
        pass

    def test_flow_forward(self):
        e = BernoulliNode(0, 'e')
        a = BernoulliNode(0, 'a', parents=[e])
        b = BernoulliNode(0, 'b', parents=[a])
        q = BernoulliNode(0, 'q', parents=[b])
        network = Network([e, a, b, q], name="Forward")
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/forward")
        self.assertSetEqual({e, a, b, q}, set(network.pruned_nodes))

    def test_flow_backward(self):
        q = BernoulliNode(0, 'q')
        b = BernoulliNode(0, 'b', parents=[q])
        a = BernoulliNode(0, 'a', parents=[b])
        e = BernoulliNode(0, 'e', parents=[a])
        network = Network([e, a, b, q], name="Backward")
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/backward")
        self.assertSetEqual({e, a, b, q}, set(network.pruned_nodes))

    def test_flow_v(self):
        a = BernoulliNode(0, 'a')
        q = BernoulliNode(0, 'q')
        e = BernoulliNode(0, 'e', parents=[a, q])
        network = Network([a, q, e], name='"V" Structure')
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/v_structure")
        self.assertSetEqual({a, q, e}, set(network.pruned_nodes))

    def test_flow_y(self):
        a = BernoulliNode(0, 'a')
        q = BernoulliNode(0, 'q')
        b = BernoulliNode(0, 'b', parents=[a, q])
        e = BernoulliNode(0, 'e', parents=[b])
        network = Network([e, a, b, q], name='"Y" Structure')
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/y_structure")
        self.assertSetEqual({a, q, b, e}, set(network.pruned_nodes))

    def test_flow_inverted_v(self):
        a = BernoulliNode(0, 'a')
        q = BernoulliNode(0, 'q', parents=[a])
        e = BernoulliNode(0, 'e', parents=[a])
        network = Network([a, q, e], name='"Inverted-V"')
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/inverted_v")
        self.assertSetEqual({a, q, e}, set(network.pruned_nodes))

    def test_basics(self):
        a = BernoulliNode(0, 'a')
        b = BernoulliNode(0, 'b', parents=[a])
        c = BernoulliNode(0, 'c', parents=[b])
        network = Network([a, b, c], name='"Blocking Forward"')
        Pruner().prune(network, {a}, {b, c}, graph_filename="test_graph_output/01_blocking_forward")
        self.assertSetEqual({a, b}, set(network.pruned_nodes))

        c = BernoulliNode(0, 'c')
        b = BernoulliNode(0, 'b', parents=[c])
        a = BernoulliNode(0, 'a', parents=[b])
        network = Network([a, b, c], name='"Blocking Backward"')
        Pruner().prune(network, {a}, {b, c}, graph_filename="test_graph_output/02_blocking_backward")
        self.assertSetEqual({a, b}, set(network.pruned_nodes))

        b = BernoulliNode(0, 'b')
        a = BernoulliNode(0, 'a', parents=[b])
        c = BernoulliNode(0, 'c', parents=[b])
        network = Network([a, b, c], name='"Blocking Inverted-V"')
        Pruner().prune(network, {a}, {b, c}, graph_filename="test_graph_output/03_blocking_inverted_v")
        self.assertSetEqual({a, b}, set(network.pruned_nodes))

        a = BernoulliNode(0, 'a')
        c = BernoulliNode(0, 'c')
        b = BernoulliNode(0, 'b', parents=[a, c])
        network = Network([a, b, c], name='"Flowing V"')
        Pruner().prune(network, {a}, {b, c}, graph_filename="test_graph_output/04_flowing_v")
        self.assertSetEqual({a, b, c}, set(network.pruned_nodes))

        a = BernoulliNode(0, 'a')
        c = BernoulliNode(0, 'c')
        b = BernoulliNode(0, 'b', parents=[a, c])
        network = Network([a, b, c], name='"Blocking V"')
        Pruner().prune(network, {a}, {c}, graph_filename="test_graph_output/05_blocking_v")
        self.assertSetEqual({a}, set(network.pruned_nodes))

        b = BernoulliNode(0, 'b')
        a = BernoulliNode(0, 'a', parents=[b])
        c = BernoulliNode(0, 'c', parents=[b])
        network = Network([a, b, c], name='"Flowing Inverted-V"')
        Pruner().prune(network, {a}, {c}, graph_filename="test_graph_output/06_flowing_inverted_v")
        self.assertSetEqual({a, b, c}, set(network.pruned_nodes))

        a = BernoulliNode(0, 'a')
        b = BernoulliNode(0, 'b', parents=[a])
        c = BernoulliNode(0, 'c', parents=[b])
        network = Network([a, b, c], name='"Flowing Forward"')
        Pruner().prune(network, {a}, {c}, graph_filename="test_graph_output/07_flowing_forward")
        self.assertSetEqual({a, b, c}, set(network.pruned_nodes))

        c = BernoulliNode(0, 'c')
        b = BernoulliNode(0, 'b', parents=[c])
        a = BernoulliNode(0, 'a', parents=[b])
        network = Network([a, b, c], name='"Flowing Backward"')
        Pruner().prune(network, {a}, {c}, graph_filename="test_graph_output/08_flowing_backward")
        self.assertSetEqual({a, b, c}, set(network.pruned_nodes))

        a = BernoulliNode(0, 'a')
        c = BernoulliNode(0, 'c')
        b = BernoulliNode(0, 'b', parents=[a, c])
        d = BernoulliNode(0, 'd', parents=[b])
        e = BernoulliNode(0, 'e', parents=[d])
        network = Network([a, b, c, d, e], name='"Flowing Y"')
        Pruner().prune(network, {a}, {c}, graph_filename="test_graph_output/09_flowing_y")
        self.assertSetEqual({a}, set(network.pruned_nodes))

        a = BernoulliNode(0, 'a')
        c = BernoulliNode(0, 'c')
        b = BernoulliNode(0, 'b', parents=[a, c])
        d = BernoulliNode(0, 'd', parents=[b])
        e = BernoulliNode(0, 'e', parents=[d])
        network = Network([a, b, c, d, e], name='"Blocking Y"')
        Pruner().prune(network, {a}, {c, e}, graph_filename="test_graph_output/10_blocking_y")
        self.assertSetEqual({a, b, c, d, e}, set(network.pruned_nodes))

    def test_mixed_sequences(self):
        self.fail()

    def test_multiple_evidence_nodes(self):
        self.fail()

    def test_multiple_query_nodes(self):
        self.fail()

    def test_multiple_paths_from_evidence_to_query(self):
        # test with some paths active, some not
        self.fail()

    def test_shachter_fig_3(self):
        _1 = BernoulliNode(0, '1')
        _3 = BernoulliNode(0, '3')
        _2 = BernoulliNode(0, '2', parents=[_1, _3])
        _5 = BernoulliNode(0, '5')
        _4 = BernoulliNode(0, '4', parents=[_5])
        _6 = BernoulliNode(0, '6', parents=[_3, _5])
        network = Network([_1, _2, _3, _4, _5, _6], name="Shachter Fig. 3")
        Pruner().prune(network, [_6], [_2, _5], graph_filename="test_graph_output/shachter_fig_3")
        self.assertSetEqual(set(network.pruned_nodes), {_1, _2, _3, _5, _6})

    def test_pre_process(self):
        _0 = BernoulliNode(0, '0')
        _1 = BernoulliNode(0, '1',parents=[_0])
        _3 = BernoulliNode(0, '3')
        _2 = BernoulliNode(0, '2', parents=[_1, _3])
        _5 = BernoulliNode(0, '5')
        _4 = BernoulliNode(0, '4', parents=[_5])
        _6 = BernoulliNode(0, '6', parents=[_3, _5])
        _7 = BernoulliNode(0, '7', parents=[_6])
        network = Network([_0, _1, _2, _3, _4, _5, _6, _7], name="Shachter Fig. 3")
        Pruner().prune(network, [_6], [_2, _5], graph_filename="test_graph_output/dg_fig_3")
        self.assertSetEqual(set(network.pruned_nodes), {_0, _1, _2, _3, _5, _6})

    def test_medium_networks(self):

        for i in range(3):
            num_nodes = 50
            network = self.create_graph_network(num_nodes, "Medium Network {}".format(i+1))
            num_query_nodes = int(.20 * num_nodes)
            num_evidence_nodes = int(.20 * num_nodes)

            Pruner().prune(network, [network.nodes[random.randint(0, num_nodes-1)] for i in range(num_query_nodes)],
                           [network.nodes[random.randint(0, num_nodes-1)] for i in range(num_evidence_nodes)])

            image_filename = "test_graph_output/medium_network_{:2d}-post.png".format(i)
            _log.info("Generating medium graph image '{}'....".format(image_filename))
            DotGraph(network).to_png(image_filename)


    def test_large_network(self):
        for num_nodes in [10**3]:
            self.prune_large_network(num_nodes, write_image=True)

        for num_nodes in [10**3, 10**4, 10**5, 10**6, 2*10**6, 5*10**6]: #, 10**7]:
            start = timer()
            self.prune_large_network(num_nodes)
            elapsed_time = timer() - start
            print("{} nodes = {}".format(num_nodes, elapsed_time))

    def prune_large_network(self, num_nodes, write_image=False):

        network = self.create_graph_network(num_nodes)
        num_query_nodes = int(.20 * num_nodes)
        num_evidence_nodes = int(.20 * num_nodes)

        _log.info("Nodes before pruning: {}...".format(len(network.nodes)))
        Pruner().prune(network, [network.nodes[random.randint(0, num_nodes-1)] for i in range(num_query_nodes)],
                       [network.nodes[random.randint(0, num_nodes-1)] for i in range(num_evidence_nodes)])
        _log.info("Nodes after pruning: {}".format(len(network.pruned_nodes)))

        if write_image:
            image_filename = "test_graph_output/large_network_{}-post.svg".format(num_nodes)
            _log.info("Generating graph image '{}'....".format(image_filename))
            DotGraph(network).to_svg(image_filename)
            _log.info("Done.")

    def create_tree_network(self, num_nodes):
        mean_parents_per_node = 1
        nodes = []
        progress_step = num_nodes // 10
        _log.info("Generating graph network with {} nodes...".format(num_nodes))
        for i in range(num_nodes):
            if (i+1) % progress_step == 0:
                _log.info("{}%...".format(10*(i+1)//progress_step))
            if i > 0:
                num_parents = min(i, max(1, abs(int(random.gauss(mean_parents_per_node, 0.01)))))
                parents = [nodes[random.randint(0, i-1)] for j in range(num_parents)]
            else:
                parents = []
            nodes.append(BernoulliNode(0, str(i), parents=parents))

        return Network(nodes, name="Large Tree Network")

    def create_graph_network(self, num_nodes, name="Graph Network"):

        nodes = [BernoulliNode(0, str(0))]

        progress_step = num_nodes // 10
        _log.info("Generating graph network with {} nodes...".format(num_nodes))
        for i in range(num_nodes):
            if (i+1) % progress_step == 0:
                _log.info("{}%...".format(10*(i+1)//progress_step))

            node = nodes[random.randint(0, len(nodes)-1)]
            new_node = BernoulliNode(0,str(i))
            is_parent = True if random.randint(0,1) == 1 else False
            if is_parent:
                node.connect_to_parent_node(new_node)
            else:
                new_node.connect_to_parent_node(node)

            nodes.append(new_node)

        # add some edges to make it more dense
        # TODO: make acyclic
        # for i in range(int(num_nodes * 0.25)):
        #     node_a = nodes[random.randint(0, len(nodes)-1)]
        #     node_b = nodes[random.randint(0, len(nodes)-1)]
        #     if not node_b in node_a.parents:
        #         node_b.connect_to_parent_node(node_a)

        return Network(nodes, name=name)

