import random
import math
import logger as log

"""
Functions needed by both filter.py and table.py have to go in here in order
to avoid circular imports. (Might be better to just merge everything into a
single file.) 
"""


def beacon_readings(actual_x, actual_y):
    """ Given the robot's (normally randomized) current position, give the normally randomized next position. """
    d_a = math.sqrt((-100 - actual_x) ** 2 + (100 - actual_y) ** 2)
    d_b = math.sqrt((150 - actual_x) ** 2 + (90 - actual_y) ** 2)
    r_a_ = random.normalvariate(d_a, 1)
    r_b_ = random.normalvariate(d_b, 1)
    return r_a_, r_b_


def new_robot_coordinates(old_x, old_y):
    """ Return new coordinates for the robot by moving a random distance in a random direction """
    d = random.normalvariate(5,1)
    theta = random.uniform(math.pi/5 - math.pi/36, math.pi/5 + math.pi/36)
    new_x = old_x + d * math.cos(theta)
    new_y = old_y + d * math.sin(theta)
    return new_x, new_y


def undirected_new_robot_coordinates(old_x, old_y):
    """ Return new coordinates for the robot by moving a random distance in ANY direction """
    # even if we allow for wide range of movement, filter still works well
    d = random.normalvariate(100,1)
    theta = random.uniform(0, 2*math.pi)
    new_x = old_x + d * math.cos(theta)
    new_y = old_y + d * math.sin(theta)
    return new_x, new_y
