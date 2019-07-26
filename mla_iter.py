import subprocess as s
import time

import numpy as np

from prOpt import local, sim_altdata

if __name__ == '__main__':

    rough_test = 0

    # if sim_altdata:
    #     if local:
    #         print("Simulating MLA data")
    #         for y in np.arange(11, 16, 1):
    #             for m in np.arange(1, 13, 1):
    #                 # print(["python3", "launch_test.py", str(rough_test), ' ', str(y), f'{m:02}', "1", str(i)])
    #                 # exit()
    #                 ym = str(y)+ f'{m:02}'
    #                 try:
    #                     iostat = s.call(["python3", "launch_test.py", str(rough_test), ym, "1"])
    #                     if iostat != 0:
    #                         print("*** Data simulation failed")
    #                         exit(iostat)
    #                 except:
    #                     print("Data simulation failed for ",ym)
    # #
    #     exit()

    for i in np.arange(0,10 ):

        if local:
            start = time.time()
            print("Processing PyXover series at external iteration",i)
            for y in np.arange(11, 16, 1):
                for m in np.arange(1, 13, 1):
                    # print(["python3", "launch_test.py", str(rough_test), ' ', str(y), f'{m:02}', "1", str(i)])
                    # exit()
                    ym = str(y)+ f'{m:02}'
                    iostat = s.call(["python3", "launch_test.py", str(rough_test), ym, "1", str(i)])
                    if iostat != 0:
                        print("*** PyGeoloc failed on iter", i)
                        # exit(iostat)
            # stop clock and print runtime
            # -----------------------------
            end = time.time()
            print('----- Runtime PyGeoloc tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
            start = time.time()

            for ymc in np.arange(0, 15, 1):
                iostat = s.call(["python3", "launch_test.py", str(rough_test), str(ymc), "2", str(i)])
                if iostat != 0:
                    print("*** PyXover failed on iter", i)
                    # exit(iostat)
            # stop clock and print runtime
            # -----------------------------
            end = time.time()
            print('----- Runtime PyXover tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
            start = time.time()

            iostat = s.call(["python3", "launch_test.py", str(rough_test), "0", "3", str(i)])
            if iostat != 0:
                print("*** PyAccum failed on iter", i)
                    # exit(iostat)
            # stop clock and print runtime
            # -----------------------------
            end = time.time()
            print('----- Runtime AccumXov tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

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

            for ymc in np.arange(0, 15, 1):
                    # print(('').join(
                    #     ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i)]))
                    loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rough_test), ' ', str(ymc), ' 2 ', str(i), '\n']))

            loadfile.close()

            print("Processing PyXover series at external iteration",i)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadPyAltSim', 'PyAltSim'])
            if iostat != 0:
                print("*** PyGeoloc failed on iter", i)
                exit(iostat)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadPyXover', 'PyXover'])
            if iostat != 0:
                print("*** PyXover failed on iter", i)
                exit(iostat)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadAccSol', 'PyAccum'])
            if iostat != 0:
                print("*** PyAccum failed on iter", i)
                exit(iostat)


