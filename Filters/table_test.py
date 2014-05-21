from table import *
from model import *

if __name__ == '__main__':

    table = Table.initialized_with_start_position(100, 0, 0)
    print(table)
    print("Status: " + table.status_string_for_output())

    print(FileModel("test_data.txt"), RobotModel(10), BeaconModel(13))
