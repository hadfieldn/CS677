import numpy
import robot
import logging
import copy
import random

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
            table.points.append(Point(*robot.beacon_readings(x,y)))
        return table

    def assign_weights(self, r_A, r_B):
        """Assign weights based on a normal distribution of likelihood from the beacon readings."""
        logging.debug("Assigning weights...")

    def create_sample_table(self):
        """Create a new sample table based on the weights of this table."""
        new_table = Table()
        print "start of table creation"
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