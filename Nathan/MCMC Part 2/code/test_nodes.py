from unittest import TestCase
from nodes import *


class TestBernoulliNode(TestCase):
    def setUp(self):
        self.b = BernoulliNode(name='B', prob=[0.001])
        self.e = BernoulliNode(name='E', prob=[0.002])
        self.a = BernoulliNode(name='A', parents=[self.b, self.e], prob=[0.95, 0.94, 0.29, 0.001])
        self.j = BernoulliNode(name='J', parents=[self.a], prob=[0.90, 0.05])
        self.m = BernoulliNode(name='M', parents=[self.a], prob=[0.70, 0.01])

    def test_probability_of_event(self):
        self.assertEquals(0.95, self.a._probability_of_event({self.b: True, self.e: True}),
                          "Incorrect probability lookup.")
        self.assertEquals(0.94, self.a._probability_of_event({self.b: True, self.e: False}),
                          "Incorrect probability lookup.")
        self.assertEquals(0.29, self.a._probability_of_event({self.b: False, self.e: True}),
                          "Incorrect probability lookup.")
        self.assertEquals(0.001, self.a._probability_of_event({self.b: False, self.e: False}),
                          "Incorrect probability lookup.")
        self.assertEquals(0.001, self.b._probability_of_event({}),
                          "Incorrect probability lookup.")
        self.assertEquals(0.001, self.b._probability_of_event({self.b: False, self.e: False}),
                          "Incorrect probability lookup.")

    def test_current_conditional_probability(self):
        self.b.current_value = True
        self.e.current_value = True
        self.a.current_value = True
        self.assertEquals(0.95, self.a.current_conditional_probability(),
                          "Incorrect conditional probability given current values of node and its parents.")

        self.e.current_value = False
        self.assertEquals(0.94, self.a.current_conditional_probability(),
                          "Incorrect conditional probability given current values of node and its parents.")

        self.a.current_value = False
        self.assertEquals(1 - 0.94, self.a.current_conditional_probability(),
                          "Incorrect conditional probability given current values of node and its parents.")

    def test_current_unnormalized_mb_probability(self):
        # initially all nodes are True
        # 0.95*0.90*0.70 = 0.5985
        self.assertAlmostEqual(0.95 * 0.90 * 0.70, self.a.current_unnormalized_mb_probability(), places=10)

        self.b.current_value = False
        self.e.current_value = False
        self.a.current_value = True
        self.j.current_value = False
        self.m.current_value = False
        # 0.001*(1-0.90)*(1-0.70) = 0.00003
        self.assertAlmostEqual(0.001 * (1 - 0.90) * (1 - 0.70), self.a.current_unnormalized_mb_probability(), places=10)

        self.a.current_value = False
        # (1-0.001)*(1-0.05)*(1-0.01) = 0.9395595
        self.assertAlmostEqual((1 - 0.001) * (1 - 0.05) * (1 - 0.01), self.a.current_unnormalized_mb_probability(),
                               places=10)

        self.b.current_value = True
        self.e.current_value = False
        self.a.current_value = True
        self.j.current_value = False
        self.m.current_value = False
        # 0.001*0.94 = 0.00094
        self.assertAlmostEqual(0.001 * 0.94, self.b.current_unnormalized_mb_probability(), places=10)

        self.b.current_value = False
        # (1-0.001)*0.001 = 0.000999
        self.assertAlmostEqual((1 - 0.001) * 0.001, self.b.current_unnormalized_mb_probability(), places=10)

        self.b.current_value = False
        self.e.current_value = False
        self.a.current_value = False
        self.j.current_value = False
        self.m.current_value = False
        # (1-0.001)*(1-0.001) = 0.998001
        self.assertAlmostEqual((1 - 0.001) * (1 - 0.001), self.b.current_unnormalized_mb_probability(), places=10)

        self.b.current_value = True
        self.e.current_value = False
        self.a.current_value = False
        self.j.current_value = False
        self.m.current_value = False
        # 0.001*(1-0.94) = 0.0006
        self.assertAlmostEqual(0.001 * (1 - 0.94), self.b.current_unnormalized_mb_probability(), places=10)


def test_probability_of_current_value_given_other_nodes(self):
    # initially all nodes are True
    p_b_true = 0.95 * 0.001
    p_b_false = 0.29 * (1 - 0.001)
    p = p_b_true / (p_b_true + p_b_false)
    self.assertAlmostEqual(p, self.b.probability_of_current_value_given_other_nodes(), places=1)

    p_a_true = 0.95 * 0.90 * 0.70
    p_a_false = (1 - 0.95) * 0.05 * 0.01
    p = p_a_true / (p_a_true + p_a_false)
    self.assertAlmostEqual(p, self.a.probability_of_current_value_given_other_nodes(), places=10)


class TestNormalNodeNode(TestCase):
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
        # f[value_, mean_, var_] := N[PDF[NormalDistribution[mean, Sqrt[var]], value]]

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

