import numpy
import robot
import logger as log
import copy
import random
import math
import scipy.stats as stats


d_C = math.sqrt(250**2 + 100)

def dist_to_coord(r_A, r_B):
    # takes distance measurements from the beacons and turns them into xy-coordinates
    alpha = math.acos((d_C**2 + r_A**2 - r_B**2)/(2*r_A*d_C))
    c = math.pi - math.atan2(1,25) - alpha
    x = r_A*math.sin(c) - 100
    y = 100 + r_A*math.cos(c)
    return x, y


class Table:


    def __init__(self):
        log.trace("Created a Table")
        self.points = []
        return

    def __str__(self):
        """String showing current state of the table, for debugging purposes"""
        return "Points:\n{}\n\nCount: {}\nMean: {}\nVariance: {}\n"\
            .format("\n".join(map(str, self.points)),
                    len(self.points),
                    ", ".join(map(str, self.means())),
                    ", ".join(map(str, self.variances())))

    @classmethod
    def initialized_with_start_position(cls, num_samples, x, y):
        """
        Start robot at (x_0,y_0) where x_0, y_0 are random variables with N(0,1).
        """
        table = Table()
        for i in range(num_samples):
            x = random.normalvariate(0, 1)
            y = random.normalvariate(0, 1)
            table.points.append(Point(x, y))
        return table

    def assign_weights(self, r_a, r_b):
        """Assign weights based on likelihood of the point given the beacon readings."""
        for point in self.points:
            d_a = math.sqrt((-100-point.x)**2 + (100-point.y)**2)
            d_b = math.sqrt((150-point.x)**2 + (90-point.y)**2)

            # the weight for (x,y) is the likelihood of getting the given
            # beacon readings (r_a, r_b) with a normal distribution centered around (d_a, d_b)

            # weight_a = stats.norm.pdf(r_a, loc=d_a, scale=1)
            # weight_b = stats.norm.pdf(r_b, loc=d_b, scale=1)

            weight_a = 1/(math.sqrt(2*math.pi))*math.exp(-0.5*(r_a - d_a)**2)
            weight_b = 1/(math.sqrt(2*math.pi))*math.exp(-0.5*(r_b - d_b)**2)
            point.w = weight_a * weight_b


    def create_sample_table(self):
        """
        Create a new sample table based on the weights of this table.

        Essentially we are filtering out points that seem unlikely. The means of
        x and y in the new table will be our most likely new position.

        The idea is to take random samples from the current distribution,
        with preference according to their weights.

        Algorithm:
        Construct a cumulative weight vector where each value is the sum
        of previous weights and the current weight. Then choose a random value
        between 0 and the sum. The selected point is the one corresponding to the
        entry in the cumulative vector that is closest to the random value.

        """
        new_table = Table()

        num_points = len(self.points)

        # construct cumulative weight vector
        cum_weights = []
        for i in range(num_points):
            if i == 0:
                cum_weights.append(self.points[i].w)
            else:
                cum_weights.append(cum_weights[i-1] + self.points[i].w)

        # randomly select points from this table to include in new
        # table, with probability of selection proportional to their weights
        for i in range(num_points):         # for each entry in new table
            random_val = random.uniform(0, cum_weights[-1])
            for j in range(num_points):     # select randomly from the current table
                if random_val <= cum_weights[j]:
                    log.info("Selected point {}".format(j))
                    break      # found it; we selected points[j]
            else:
                # no match found -- this should never happen
                log.error("INTERNAL ERROR: No cumulative weight entry could be found for the random selection.")

            # do a deep copy because we're sampling with replacement
            new_table.add_point_to_table(self.points[j])

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
        return " & ".join(map(str, (self.means() + self.variances()))) + " \\\\"

    def transition(self):
        """
        Assume the x, y means of this sample are the actual robot coordinates,
        move all the points to their relative positions about this new origin.
        """
        new_x, new_y = self.means()
        for point in self.points:
            point.transition(new_x, new_y)

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
        return "({}, {})[w={}]".format( self.x, self.y, self.w)

    def transition(self, new_x, new_y ):
        """Move point to new origin."""
        self.x -= new_x
        self.y -= new_y
        return

