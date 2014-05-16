import table
from table import *
import random as rand
import math

def get_reading(robot_x, robot_y):
    d_A = math.sqrt((-100-robot_x)**2+(100-robot_y)**2)
    d_B = math.sqrt((150-robot_x)**2+(90-robot_y)**2)
    r_A = rand.normalvariate(d_A, 1)
    r_B = rand.normalvariate(d_B, 1)
    return r_A, r_B

def update_robot(robot_x, robot_y):
    robot_x = robot_x
    robot_y = robot_y

def transition():
    print()

def append_to_file(means, variances, robot_x, robot_y):
    return

if __name__ == '__main__':
    file = open("output.txt", "w")

    robot_x = rand.normalvariate(0, 1)
    robot_y = rand.normalvariate(0, 1)
    current_table = Table()
    current_table.init_with_start_position(robot_x, robot_y)
    for i in range(10):
        r_A, r_B = get_reading(robot_x, robot_y)
        current_table.assign_weights(r_A, r_B)
        next_table = current_table.create_sample_table()
        next_table.append_status_to_file(file)
        next_table.transition()
        current_table = next_table
        update_robot(robot_x, robot_y)
    file.close()




