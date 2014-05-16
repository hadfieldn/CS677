__author__ = 'Daniel'

class Table:

    def __init__(self):
        self.points = []
        print("created a Table")

    def init_with_start_position(self,x,y):
        print()

    def assign_weights(self, r_A, r_B):
        print()

    def create_sample_table(self):
        return Table()

    def get_means(self):
        return 0

    def get_variances(self):
        return 0

    def append_status_to_file(self, file):
        return

    def transition(self):
        for point in self.points:
            point.transition()




class Point:
    x = 0.
    y = 0.
    w = 0.
    def __init__(self, x_, y_, w_=0):
        self.x = x_
        self.y = y_
        self.w = w_

    def transition(self):
        return
