from nodes import *
from network import *
import logging

log = logging.getLogger("mcmc")

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
logging.getLogger().setLevel(logging.ERROR)
# logging.getLogger("nodes").setLevel(logging.ERROR)
# logging.getLogger("network").setLevel(logging.ERROR)
# logging.getLogger("mcmc").setLevel(logging.ERROR)


b = BernoulliNode(name='B', prob=[.3], value=False)
k = BernoulliNode(name='K', prob=[.5, .3, .3, .2], value=True)
m = BernoulliNode(name='M', prob=[.14, .3], value=True)
h = BernoulliNode(name='H', prob=[.6], value=True)
s = BernoulliNode(name='S', prob=[.05, .1, .1, .5, .1, .4, .5, .8], value=True)

b.children = [k, s]
k.children = [s]
k.parents = [b, m]
m.parents = [h]
m.children = [k]
h.children = [m, s]
s.parents = [b, k, h]

h.is_observed = True
m.is_observed = True

network = Network(nodes=[b, k, m, h, s])
samples = network.collect_samples(burn=3000, n=100000)
log.info("Totals: " + str(samples.totals()))
print("P(S=True | H=True, M=True) = " + str(samples.p({s: True}, {h: True, m: True})))

#samples.plot_mixing("P(S=True | H=True, M=True)", {s: True}, {h: True, m: True})
