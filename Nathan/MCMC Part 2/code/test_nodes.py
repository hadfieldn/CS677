from unittest import TestCase
from node_bernoulli import *
from node_normal import *
from node_invgamma import *
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')

class TestBernoulliNode(TestCase):

    def setUp(self):
        self.b = BernoulliNode(name='B', p=[0.001])
        self.e = BernoulliNode(name='E', p=[0.002])
        self.a = BernoulliNode(name='A', parents=[self.b, self.e], p=[0.95, 0.94, 0.29, 0.001])
        self.j = BernoulliNode(name='J', parents=[self.a], p=[0.90, 0.05])
        self.m = BernoulliNode(name='M', parents=[self.a], p=[0.70, 0.01])

    def test_probability_of_event(self):
        self.assertEquals(0.95, self.a.probability_of_event({self.b: True, self.e: True}))
        self.assertEquals(0.94, self.a.probability_of_event({self.b: True, self.e: False}))
        self.assertEquals(0.29, self.a.probability_of_event({self.b: False, self.e: True}))
        self.assertEquals(0.001, self.a.probability_of_event({self.b: False, self.e: False}))
        self.assertEquals(0.001, self.b.probability_of_event({}))
        self.assertEquals(0.001, self.b.probability_of_event({self.b: False, self.e: False}))


class TestNormalNode(TestCase):
    def setUp(self):
        pass

    def test_str(self):
        n = NormalNode(0.1, name='N', mean=0, var=1)
        self.assertEquals("N(0, 1) = 0.1", str(n))

        m = NormalNode(0.1)
        self.assertEquals("NormalNode(0, 1) = 0.1", str(m))

        o = NormalNode(0.1, name='O', mean=n, var=m)
        self.assertEquals("O(N, NormalNode) = 0.1", str(o))

    def test_pdf_name(self):
        self.assertEquals("N(0, 1)", NormalNode(0, 'N', 0, 1).pdf_name)

    def test_log_current_conditional_probability(self):
        # test values generated using Mathematica function
        # p[value_, mean_, var_] := N[PDF[NormalDistribution[mean, Sqrt[var]], value]]

        n = NormalNode(0, name='N', mean=0, var=1)
        self.assertAlmostEquals(0.398942, n.current_conditional_probability(), delta=0.0001)

        n = NormalNode(0, name='N', mean=1, var=3)
        self.assertAlmostEquals(0.19497, n.current_conditional_probability(), delta=0.0001)

        n = NormalNode(4, name='N', mean=1, var=3)
        self.assertAlmostEquals(0.0513934, n.current_conditional_probability(), delta=0.0001)

        n = NormalNode(12.4, name='N', mean=6.8, var=8.3)
        self.assertAlmostEquals(0.0209373, n.current_conditional_probability(), delta=0.0001)

    def test_log_current_unnormalized_mb_probability(self):
        n1 = NormalNode(0.3, name='N1', mean=0, var=1)
        n2 = NormalNode(0.4, name='N2', mean=n1, var=1)

        self.assertAlmostEquals(0.396953, n2.current_unnormalized_mb_probability(), delta=0.001)
        self.assertAlmostEquals(0.151393, n1.current_unnormalized_mb_probability(), delta=0.001)


class TestInvGammaNode(TestCase):
    def setUp(self):
        pass

    def test_log_current_conditional_probability(self):
        # Test values generated using Mathematica function
        # p[value_, shape_, scale_] := N[PDF[InverseGammaDistribution[shape, scale], value]]
        #
        # Plot[{p[x, 1, 1], p[x, 2, 1], p[x, 3, 1], p[x, 3, 0.5]}, {x, 0, 3}, PlotRange -> All, Filling -> Axis]

        n = InvGammaNode(1, name='IGamma', shape=1)
        self.assertAlmostEquals(0.367879, n.current_conditional_probability(), delta=0.0001)

        n = InvGammaNode(1, name='IGamma', shape=2)
        self.assertAlmostEquals(0.367879, n.current_conditional_probability(), delta=0.0001)

        n = InvGammaNode(1, name='IGamma', shape=3)
        self.assertAlmostEquals(0.18394, n.current_conditional_probability(), delta=0.0001)

        n = InvGammaNode(2.5, name='IGamma', shape=3, scale=0.5)
        self.assertAlmostEquals(0.00130997, n.current_conditional_probability(), delta=0.0001)

