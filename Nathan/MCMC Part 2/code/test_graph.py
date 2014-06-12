from unittest import TestCase
from network import *
from graph import *
from node_normal import *
from node_bernoulli import *
from node_beta import *
from node_gamma import *
from node_poisson import *
import textwrap

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


        self.fail("Have to check this test manually.")


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
        self.fail()

    def test_flow_backward(self):
        q = BernoulliNode(0, 'q')
        b = BernoulliNode(0, 'b', parents=[q])
        a = BernoulliNode(0, 'a', parents=[b])
        e = BernoulliNode(0, 'e', parents=[a])
        network = Network([e, a, b, q], name="Backward")
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/backward")
        self.fail()

    def test_flow_v(self):
        a = BernoulliNode(0, 'a')
        q = BernoulliNode(0, 'q')
        e = BernoulliNode(0, 'e', parents=[a, q])
        network = Network([a, q, e], name='"V" Structure')
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/v_structure")
        self.fail()

    def test_flow_y(self):
        a = BernoulliNode(0, 'a')
        q = BernoulliNode(0, 'q')
        b = BernoulliNode(0, 'b', parents=[a, q])
        e = BernoulliNode(0, 'e', parents=[b])
        network = Network([e, a, b, q], name='"Y" Structure')
        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/y_structure")
        self.fail()

    def test_flow_inverted_v(self):
        a = BernoulliNode(0, 'a')
        q = BernoulliNode(0, 'q', parents=[a])
        e = BernoulliNode(0, 'e', parents=[a])
        network = Network([a, q, e], name='"Inverted-V"')

        Pruner().prune(network, {q}, {e}, graph_filename="test_graph_output/inverted_v")
        self.fail()

    def test_mixed_sequences(self):
        self.fail()

    def test_multiple_evidence_nodes(self):
        self.fail()

    def test_multiple_query_nodes(self):
        self.fail()

    def test_multiple_paths_from_evidence_to_query(self):
        # test with some paths active, some not
        self.fail()