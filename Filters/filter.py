from table import *
import logger as log
import model

if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(module)s %(funcName)s(): %(message)s')
    log.set_level(log.ERROR)            # use our own logger to ensure compatibility

    file = open("output.txt", "w")

    current_table = Table.initialized_with_start_position(100, 0, 0)
#    current_table = Table.initialized_with_start_position(1000, *robot.new_robot_coordinates(0, 0))
    log.debug("Initial sample table:\n{}".format(current_table))

    #model = model.BeaconModel(10)              # (nh) gives beacon coordinates (what I think we need to use)
    model = model.FileModel("test_data.txt")   # (nh) gives beacon coordinates (test data from Dr. Seppi)

    print(model)

    for r_a, r_b in model.coordinates():

        log.debug("Obtained beacon coordinates ({}, {}), determining likely robot position...".format(r_a, r_b))

        current_table.transition()
        log.debug("After transitioning:\n{}".format(current_table))

        current_table.assign_weights(r_a, r_b)
        log.debug("Weighted samples:\n{}\n".format(current_table))

        next_table = current_table.create_sample_table()
        log.debug("New sample table:\n{}\n".format(next_table))


        status = next_table.status_string_for_output()
        file.write(status + "\n")
        print(status)


        current_table = next_table
    file.close()




