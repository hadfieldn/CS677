from __future__ import generators
import logger as log
import random
import robot

class Model(object):

    def __init__(self, coords):
        self.coords = coords
    def __str__(self):
        """Show representation of model for debugging purposes."""
        coord_str = "\n".join(map(lambda coord: ", ".join(map(str, coord)), self.coords))
        return "Model coordinates ({}) [robot_x, robot_y, beacon_a, beacon_b]:\n{}\n\n".format(len(self.coords), coord_str)


    def coordinates(self):
        """Generator for iterating through coordinates as [x, y] pairs."""
        for coord in self.coords:
            yield [coord[2], coord[3]]


class BeaconModel(Model):
    """
    Generate beacon coordinates at runtime.
    This model cannot currently be used because Table.assign_weights
    assumes robot coordinates instead of beacon coordinates.
    """

    def __init__(self, count):
        super(BeaconModel, self).__init__(self.generate_coordinates(count))

    def generate_coordinates(self, count):
        """Generate beacon coordinates based on random movement of robot."""
        coordinates = []

        actual_x = random.normalvariate(0, 1)
        actual_y = random.normalvariate(0, 1)

        x = y = 0.0
        for i in range(count):
            # move the robot and generate new beacon readings with random
            # offsets from actual robot coordinates (to simulate noise)
            beacon_x, beacon_y = robot.beacon_readings(actual_x, actual_y)
            coordinates.append([actual_x, actual_y, beacon_x, beacon_y])
            actual_x, actual_y = robot.new_robot_coordinates(actual_x, actual_y)

        return coordinates


class FileModel(Model):
    """
    Load test readings from a file. Assumes lines are of this format:

        6.23506410875 & 3.47428776487 & 143.69345025 & 166.824055471 \\

    where the first two columns are ignored and the second two columns are
    the coordinates to be used.
    """

    def __init__(self, file_name):
        super(FileModel, self).__init__(self.read_coordinates(file_name))

    def read_coordinates(self, file_name):
        file = open(file_name, "r")
        coordinates = []
        log.debug("Importing coordinates from file {}...".format(file_name))
        for line in file:
            if "\\\\" in line:
                values = line.replace("\\\\", "").split("&")
                # TODO: (nh) I think we should be using columns 3 and 4 ("the input data")
                # Assignment states: "Use the noisy input values given above [i.e.,
                # columns 3 and 4] (NOT THE TRUE VALUES GIVEN ABOVE! They are included
                # for reference only!) to infer the location of the robot"

                # cols 0, 1 are actual robot positions
                # coordinates.append([float(values[0].strip()), float(values[1].strip())])

                # cols 2, 3 are beacon readings
                log.debug(line)
                coordinate = [float(values[0].strip()), float(values[1].strip()), float(values[2].strip()), float(values[3].strip())]
                coordinates.append(coordinate)
        file.close()
        return coordinates


class KalmanSampleDataModel(Model):
    """
    Load test readings from a file. Assumes lines are of this format:

        1 0.8 1.91 0.622222222222 1.48555555556

    where the second and third columns are the coordinates to import.
    """

    def __init__(self, file_name):
        super(KalmanSampleDataModel, self).__init__(self.read_coordinates(file_name))

    def read_coordinates(self, file_name):
        file = open(file_name, "r")
        coordinates = []
        log.debug("Importing coordinates from file {}...".format(file_name))
        for line in file:
            log.debug(line)
            values = line.split(" ")
            coordinate = [float(values[3].strip()), float(values[4].strip()), float(values[1].strip()), float(values[2].strip())]
            coordinates.append(coordinate)
        file.close()
        return coordinates





