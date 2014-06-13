import test_graph
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')

test_graph.TestPruner().test_large_network()
