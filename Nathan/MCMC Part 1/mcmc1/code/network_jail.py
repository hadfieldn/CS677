from nodes import *
from network import *
import logging

log = logging.getLogger("mcmc")

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
logging.getLogger().setLevel(logging.ERROR)
# logging.getLogger("nodes").setLevel(logging.ERROR)
# logging.getLogger("network").setLevel(logging.ERROR)
# logging.getLogger("mcmc").setLevel(logging.ERROR)


b = BernoulliNode(name='B', prob=[.9], value=True)
i = BernoulliNode(name='I', prob=[.9, .5, .5, .1], value=True)
m = BernoulliNode(name='M', prob=[.1], value=True)
g = BernoulliNode(name='G', prob=[.9, .8, .0, .0, .2, .1, .0, .0], value=False)
j = BernoulliNode(name='J', prob=[0.9, 0.0], value=False)

b.children = [g, i]
i.children = [g]
i.parents = [b, m]
m.children = [i, g]
g.children = [j]
g.parents = [b, i, m]
j.parents = [g]

b.is_observed = True
i.is_observed = True
m.is_observed = True

network = Network(nodes=[b, i, m, g, j])
samples = network.collect_samples(burn=3000, n=100000)
log.info("Totals: " + str(samples.totals()))
print("P(J=True | B=True, I=True, M=True) = " + str(samples.p({j: True}, {b: True, i: True, m: True})))

#samples.plot_mixing("P(J=True | B=True, I=True, M=True)", {j: True}, {b: True, i: True, m: True})
