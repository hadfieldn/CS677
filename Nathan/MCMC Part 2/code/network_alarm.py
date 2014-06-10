from node_bernoulli import *
from node_beta import *
from network import *

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
_log = logging.getLogger("alarm")

def generate_sample_data():
    b = BernoulliNode(0, name='B', p=[0.2])
    e = BernoulliNode(0, name='E', p=[0.3])
    a = BernoulliNode(0, name='A', parents=[b, e], p=[0.95, 0.94, 0.29, 0.2])
    j = BernoulliNode(0, name='J', parents=[a], p=[0.90, 0.2])
    m = BernoulliNode(0, name='M', parents=[a], p=[0.70, 0.3])

    network = Network(nodes=[b, e, a, j, m])
    samples = network.collect_samples(burn=1000, n=10000)

    file = open("alarm.dat", "w")
    samples.write_to_file(file, 100)
    file.close()

def read_sample_data():
    data = []
    for line in open("alarm1000.dat"):
        data.append(list(map(int, line.split(','))))

    return data

generate_sample_data()
data = read_sample_data()
#print("{}".format(data[0]))

p_m_a = BetaNode(0.1, "P(M|A=true)", alpha=1, beta=1)
p_m_not_a = BetaNode(0.1, "P(M|A=false)", alpha=1, beta=1)
p_j_a = BetaNode(0.1, "P(J|A=true)", alpha=1, beta=1)
p_j_not_a = BetaNode(0.1, "P(J|A=false)", alpha=1, beta=1)
p_a_b_e = BetaNode(0.1, "P(A|B=true,E=true)", alpha=1, beta=1)
p_e = BetaNode(0.1, "P(E=true)", alpha=1, beta=1)
p_b = BetaNode(0.1, "P(B=true)", alpha=1, beta=1)
p_a_b_not_e = BetaNode(0.1, "P(A|B=true,E=false)", alpha=1, beta=1)
p_a_not_b_e = BetaNode(0.1, "P(A|B=false,E=true)", alpha=1, beta=1)
p_a_not_b_not_e = BetaNode(0.1, "P(A|B=false,E=false)", alpha=1, beta=1)

for line in data:
    _log.info("Creating nodes for sample {}...".format(line))
    b = BernoulliNode(line[0], name='B', p=[p_b], observed=True)
    e = BernoulliNode(line[1], name='E', p=[p_e], observed=True)
    a = BernoulliNode(line[2], name='A', parents=[b, e], p=[p_a_b_e, p_a_b_not_e, p_a_not_b_e, p_a_not_b_not_e], observed=True)
    j = BernoulliNode(line[3], name='J', parents=[a], p=[p_j_a, p_j_not_a], observed=True)
    m = BernoulliNode(line[4], name='M', parents=[a], p=[p_m_a, p_m_not_a], observed=True)

#nodes = [p_m_a, p_m_not_a, p_j_a, p_j_not_a, p_a_b_e, p_e, p_b, p_a_b_not_e, p_a_not_b_not_e, p_a_not_b_e]
nodes = [p_a_b_not_e]
network = Network(nodes)
samples = network.collect_samples(burn=0, n=1000)

for node in nodes:
    samples.plot_node(node)
    #samples.plot_histogram_for_node(node)


# samples.plot_node(j)
# samples.plot_node(m)
# print("P(B=False | J=True, M=True) = " + str(samples.p({b: False}, {j: True, m: True})))
# print("P(B=True | J=True, M=True) = " + str(samples.p({b: True}, {j: True, m: True})))
# print("P(A=True | J=True, M=True) = " + str(samples.p({a: True}, {j: True, m: True})))
# print("P(E=True | J=True, M=True) = " + str(samples.p({e: True}, {j: True, m: True})))
#
# samples.plot_mixing("P(B=True|J=True,M=True)", {b: True}, {j: True, m: True})
# samples.plot_histogram("P(B=True|J=True,M=True)", {b: True}, {j: True, m: True})
