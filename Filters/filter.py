import random
from table import *
import robot
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')

    file = open("output.txt", "w")

    actual_x = random.normalvariate(0, 1)
    actual_y = random.normalvariate(0, 1)
    current_table = Table.initialized_with_start_position( 100, actual_x, actual_y )
    for i in range(10):
        r_a, r_b = robot.beacon_readings(actual_x, actual_y)
        current_table.assign_weights(r_a, r_b)
        next_table = current_table.create_sample_table()
        file.write(next_table.status_string_for_output())
        next_table.transition()
        current_table = next_table
        actual_x, actual_y = robot.new_robot_coordinates(actual_x, actual_y)
    file.close()




