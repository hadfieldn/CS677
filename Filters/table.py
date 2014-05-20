import numpy
import robot
import logging
import copy
import random
import math

d_C = math.sqrt(250**2 + 100)

def dist_to_coord(r_A, r_B):
    # takes distance measurements from the beacons and turns them into xy-coordinates
    alpha = math.acos((d_C**2 + r_A**2 - r_B**2)/(2*r_A*d_C))
    c = math.pi - math.atan2(1,25) - alpha
    x = r_A*math.sin(c) - 100
    y = 100 + r_A*math.cos(c)
    return x, y
    pass


class Table:

    points = []

    def __init__(self):
        logging.debug("Created a Table")

    def __str__(self):
        """String showing current state of the table, for debugging purposes"""
        return "Points:\n{}\n\nMeans: {}\nVariances: {}\n"\
            .format("\n".join(map(str,self.points)),
                    ", ".join(map(str,self.means)),
                    ", ".join(map(str,self.variances)))

    @classmethod

    def initialized_with_start_position(cls, num_points, x, y):
        """Constructs a table and initializes it with points located a random offset from given position. """
        table = Table()
        for i in range(num_points):
            r_A, r_B = robot.beacon_readings(x, y)
            guessed_x, guessed_y = dist_to_coord(r_A, r_B)
            table.points.append(Point(guessed_x, guessed_y))
        return table

    def assign_weights(self, r_A, r_B):
        """Assign weights based on a normal distribution of likelihood from the beacon readings."""
        for point in self.points:
            guessed_d_A = math.sqrt((-100-point.x)**2 + (100-point.y)**2)
            guessed_d_B = math.sqrt((-150-point.y)**2 + (90-point.y)**2)
            weight_A = 1/(math.sqrt(2*math.pi))*math.exp(-0.5*(r_A - guessed_d_A)**2)
            weight_B = 1/(math.sqrt(2*math.pi))*math.exp(-0.5*(r_B - guessed_d_B)**2)
            point.w = weight_A*weight_B

    def create_sample_table(self):
        """Create a new sample table based on the weights of this table."""
        new_table = Table()
        for i in range(len(self.points)):
            random_weight = random.random()
            for point in self.points:
                if point.get_normalized_weight() < random_weight:
                    new_table.add_point_to_table(point)
                    break
        return new_table

    def means(self):
        mean_x = numpy.mean([point.x for point in self.points])
        mean_y = numpy.mean([point.y for point in self.points])
        return mean_x, mean_y

    def variances(self):
        var_x = numpy.var([point.x for point in self.points])
        var_y = numpy.var([point.y for point in self.points])
        return var_x, var_y

    def status_string_for_output(self):
        return " & ".join(map(str, (self.means() + self.variances()))) + " \\\\\n"

    def transition(self):
        """Apply random robot movements to each point."""
        for point in self.points:
            point.transition()

    def add_point_to_table(self, point_to_add):
        # makes a copy of a point object and adds it to the table
        self.points.append(copy.deepcopy(point_to_add))


class Point:
    x = 0.
    y = 0.
    w = 0.
    def __init__(self, x_, y_, w_=0.0):
        self.x = x_
        self.y = y_
        self.w = w_

    def __str__(self):
        return "({},{})[w={}]".format( self.x, self.y, self.w)

    def transition(self):
        """Apply random robot movement."""
        self.x, self.y = robot.new_robot_coordinates(self.x, self.y)
        return

    def get_normalized_weight(self):
        # needs to return a value from 0-1
        return self.w