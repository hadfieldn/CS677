from node_bernoulli import *
from node_beta import *
from network import *
import sys

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
_log = logging.getLogger("alarm")

def generate_sample_data():
    b = BernoulliNode(0, name='B', p=[0.001])
    e = BernoulliNode(0, name='E', p=[0.002])
    a = BernoulliNode(0, name='A', parents=[b, e], p=[0.95, 0.94, 0.29, 0.001])
    j = BernoulliNode(0, name='J', parents=[a], p=[0.90, 0.05])
    m = BernoulliNode(0, name='M', parents=[a], p=[0.70, 0.01])

    network = Network(nodes=[b, e, a, j, m])
    samples = network.collect_samples(burn=1000, n=10000, skip=100)

    file = open("alarm_orig_10000.dat", "w")
    samples.write_to_file(file)
    file.close()

def read_sample_data():
    data = []
    for line in open("alarm_orig_10000.dat"):
        data.append(list(map(int, line.split(','))))

    return data

# generate_sample_data()
# sys.exit()

data = read_sample_data()
#print("{}".format(data[0]))

p_m_a = BetaNode(0.1, "P(M|A=true)", alpha=1, beta=1)
p_m_not_a = BetaNode(0.1, "P(M|A=false)", alpha=1, beta=1)
p_j_a = BetaNode(0.1, "P(J|A=true)", alpha=1, beta=1)
p_j_not_a = BetaNode(0.1, "P(J|A=false)", alpha=1, beta=1)

# p_a_b_e = BetaNode(0.1, "P(A|B=true,E=true)", alpha=1, beta=1)
# p_e = BetaNode(0.1, "P(E=true)", alpha=1, beta=1)
# p_b = BetaNode(0.1, "P(B=true)", alpha=1, beta=1)
# p_a_b_not_e = BetaNode(0.1, "P(A|B=true,E=false)", alpha=1, beta=1)
# p_a_not_b_e = BetaNode(0.1, "P(A|B=false,E=true)", alpha=1, beta=1)
# p_a_not_b_not_e = BetaNode(0.1, "P(A|B=false,E=false)", alpha=1, beta=1)

for line in data:
    _log.info("Creating nodes for sample {}...".format(line))
    a = b = e = a = j = m = None
    if line[0] != -1:
        b = BernoulliNode(line[0], name='B', p=[0.001], observed=True)
    if line[1] != -1:
        e = BernoulliNode(line[1], name='E', p=[0.002], observed=True)
    if line[2] != -1 and b is not None and e is not None:
        a = BernoulliNode(line[2], name='A', parents=[b, e], p=[0.95, 0.94, 0.29, 0.001], observed=True)
    if line[3] != -1 and a is not None:
        j = BernoulliNode(line[3], name='J', parents=[a], p=[p_j_a, p_j_not_a], observed=True)
    if line[4] != -1 and a is not None:
        m = BernoulliNode(line[4], name='M', parents=[a], p=[p_m_a, p_m_not_a], observed=True)

nodes = [p_m_a, p_m_not_a, p_j_a, p_j_not_a]
#nodes = [p_a_b_not_e]
network = Network(nodes)
samples = network.collect_samples(burn=1000, n=1000)

# for node in nodes:
#     node.is_observed = True
#
# b = BernoulliNode(1, name='B', p=[p_b], observed=False)
# e = BernoulliNode(1, name='E', p=[p_e], observed=False)
# a = BernoulliNode(1, name='A', parents=[b, e], p=[p_a_b_e, p_a_b_not_e, p_a_not_b_e, p_a_not_b_not_e], observed=False)
# j = BernoulliNode(1, name='J', parents=[a], p=[p_j_a, p_j_not_a], observed=False)
# m = BernoulliNode(1, name='M', parents=[a], p=[p_m_a, p_m_not_a], observed=True)
#
# # network.add_nodes([b, e, a, j, m])
# _log.info("Collecting samples after making m observed")
# samples = network.collect_samples(burn=100, n=5000)
# print("P(B=True|M=True) = {}".format(samples.p({b: 1}, {m: 1})))
# samples.plot_node(b)

for node in nodes:
    samples.plot_node(node)
    samples.plot_histogram_for_node(node)


