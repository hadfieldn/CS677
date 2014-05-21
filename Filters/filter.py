import random
from table import *
import robot
import logger as log
import model

if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
    log.set_level(log.DEBUG)            # use our own logger to ensure compatibility

    file = open("output.txt", "w")

    actual_x = random.normalvariate(0, 1)
    actual_y = random.normalvariate(0, 1)
    current_table = Table.initialized_with_start_position(100, actual_x, actual_y)
    model = model.RobotModel(10)                # (nh) gives robot coordinates
    #model = model.BeaconModel(10)              # (nh) gives beacon coordinates (what I think we need to use)
    #model = model.FileModel("test_data.txt")   # (nh) gives beacon coordinates (test data from Dr. Seppi)
    for r_a, r_b in model.coordinates():
        log.debug("Coordinate: ({}, {})".format(r_a, r_b))
        current_table.assign_weights(r_a, r_b)
        next_table = current_table.create_sample_table()

        if log.is_debug_enabled():
            log.debug("New Weight table:\n{}".format(next_table))

        status = next_table.status_string_for_output()
        file.write(status)
        print(status)

        next_table.transition()
        current_table = next_table
    file.close()




