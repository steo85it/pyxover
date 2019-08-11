import subprocess as s

import numpy as np

from prOpt import local

if __name__ == '__main__':

    rough_test = 0

    for i in np.arange(0,1):

        if local:
            print("Processing PyXover series at external iteration",i)
            iostat = s.call(["python3", "launch_test.py", str(rough_test), "1301", "1", str(i)])
            if iostat != 0:
                print("*** PyGeoloc failed on iter", i)
                exit(iostat)
            iostat = s.call(["python3", "launch_test.py", str(rough_test), "9", "2", str(i)])
            if iostat != 0:
                print("*** PyXover failed on iter", i)
                exit(iostat)
            iostat = s.call(["python3", "launch_test.py", str(rough_test), "9", "3", str(i)])
            if iostat != 0:
                print("*** PyAccum failed on iter", i)
                exit(iostat)
        else:

            loadfile = open("loadPyGeoloc", "w")  # write mode

            for y in np.arange(11, 16, 1):
                for m in np.arange(1, 13, 1):
                    # print(('').join(
                    #     ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i)]))
                    loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i), '\n']))

            loadfile.close()

            loadfile = open("loadPyXover", "w")  # write mode

            for ymc in np.arange(0, 16, 1):
                    loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rough_test), ' ', str(ymc), ' 2 ', str(i), '\n']))

            loadfile.close()

            print("Processing PyXover series at external iteration",i)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadPyGeoloc', 'PyAltSim', '8', '99:99:99', '10'])
            if iostat != 0:
                print("*** PyGeoloc failed on iter", i)
                exit(iostat)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadPyXover', 'PyXover', '8', '99:99:99', '10'])
            if iostat != 0:
                print("*** PyXover failed on iter", i)
                exit(iostat)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadAccSol', 'PyAccum', '8', '99:99:99', '10'])
            if iostat != 0:
                print("*** PyAccum failed on iter", i)
                exit(iostat)