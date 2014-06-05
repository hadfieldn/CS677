from node import *
from network import *
import logging

log = logging.getLogger("mcmc")

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
logging.getLogger().setLevel(logging.ERROR)
# logging.getLogger("nodes").setLevel(logging.ERROR)
# logging.getLogger("network").setLevel(logging.ERROR)
# logging.getLogger("mcmc").setLevel(logging.ERROR)


b = BernoulliNode(value=False, name='B', prob=[0.001])
e = BernoulliNode(value=False, name='E', prob=[0.002])
a = BernoulliNode(value=False, name='A', parents=[b, e], prob=[0.95, 0.94, 0.29, 0.001])
j = BernoulliNode(value=True, name='J', parents=[a], prob=[0.90, 0.05], observed=True)
m = BernoulliNode(value=True, name='M', parents=[a], prob=[0.70, 0.01], observed=True)

network = Network(nodes=[b, e, a, j, m])
samples = network.collect_samples(burn=0, n=10000)
log.info("Totals: " + str(samples.totals()))
print("P(B=False | J=True, M=True) = " + str(samples.p({b: False}, {j: True, m: True})))
print("P(B=True | J=True, M=True) = " + str(samples.p({b: True}, {j: True, m: True})))
print("P(A=True | J=True, M=True) = " + str(samples.p({a: True}, {j: True, m: True})))
print("P(E=True | J=True, M=True) = " + str(samples.p({e: True}, {j: True, m: True})))

samples.plot_mixing("P(B=True|J=True,M=True)", {b: True}, {j: True, m: True})
samples.plot_histogram("P(B=True|J=True,M=True)", {b: True}, {j: True, m: True})


# Different observed nodes; have to resample

# network = Network(nodes=[b, e, a, j, m])
# samples = network.collect_samples(burn=10000, n=100000)
#
# j.current_value = False
# m.current_value = False
# samples = network.collect_samples(burn=10000, n=100000)
# print("P(B=True | J=False, M=False) = " + str(samples.p({b: True}, {j: False, m: False})))
# j.current_value = True
# m.current_value = False
# samples = network.collect_samples(burn=10000, n=100000)
# print("P(B=True | J=True, M=False) = " + str(samples.p({b: True}, {j: True, m: False})))
# j.is_observed = True
# j.current_value = True
# m.is_observed = False
# samples = network.collect_samples(burn=10000, n=100000)
# print("P(B=True | J=True) = " + str(samples.p({b: True}, {j: True})))
# j.is_observed = False
# m.is_observed = True
# m.current_value = True
# samples = network.collect_samples(burn=10000, n=100000)
# print("P(B=True | M=True) = " + str(samples.p({b: True}, {m: True})))
#

