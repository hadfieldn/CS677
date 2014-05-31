from unittest import TestCase
from network import *
from nodes import *
import textwrap

class TestSamplesProcessor(TestCase):

    def setUp(self):
        self.a = BernoulliNode(name='A', prob=[0.5])
        self.b = BernoulliNode(name='B', prob=[0.2])
        self.c = BernoulliNode(name='C', prob=[0.7])
        self.samples = [(True, False, False),
                        (True, False, True),
                        (False, False, True),
                        (False, True, True),
                        (True, False, True)]
        self.processor = SamplesProcessor([self.a, self.b, self.c], self.samples)

    def test_str(self):
        samples_str = textwrap.dedent("""\
            A, B, C
            True, False, False
            True, False, True
            False, False, True
            False, True, True
            True, False, True""")
        self.assertEqual(samples_str, str(self.processor), "Incorrect string representation")

    def test_is_sample_match(self):

        self.assertTrue(self.processor.is_sample_match([True, False, False], {self.a: True, self.b: False, self.c: False}))
        self.assertFalse(self.processor.is_sample_match([True, False, False], {self.a: True, self.b: False, self.c: True}))
        self.assertTrue(self.processor.is_sample_match([True, False, False], {self.a: True, self.c: False}))
        self.assertFalse(self.processor.is_sample_match([True, False, False], {self.a: True, self.c: True}))
        self.assertFalse(self.processor.is_sample_match([True, False, False], {self.a: True, self.c: True}))
        self.assertTrue(self.processor.is_sample_match([True, False, False], {self.a: True}))
        self.assertTrue(self.processor.is_sample_match([True, False, False], {}))

    def test_p(self):
        self.assertEquals(1/4, self.processor.p({self.a: False, self.b: True}, {self.c: True}))
        self.assertEquals(2/3, self.processor.p({self.a: True}, {self.b: False, self.c: True}))
        self.assertEquals(3/5, self.processor.p({self.a: True}, {}))
