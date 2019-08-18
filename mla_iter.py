import subprocess as s
import time

import numpy as np

from prOpt import local, sim_altdata

if __name__ == '__main__':

    rough_test = 0

    for i in np.arange(0, 1 ):

        if local:
            start = time.time()
            print("Processing PyXover series at external iteration",i)
            for y in np.append([8],np.arange(11, 16, 1)):
                for m in np.arange(1, 13, 1):
                    # print(["python3", "launch_test.py", str(rough_test), ' ', str(y), f'{m:02}', "1", str(i)])
                    # exit()
                    ym = f'{y:02}'+ f'{m:02}'
                    iostat = s.call(["python3", "launch_test.py", str(rough_test), ym, "1", str(i)])
                    if iostat != 0:
                        print("*** PyGeoloc failed on iter", i)
                        # exit(iostat)
            # stop clock and print runtime
            # -----------------------------
            end = time.time()
            print('----- Runtime PyGeoloc tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
            start = time.time()

            for ymc in np.arange(0, 21, 1):
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
            start = time.time()
            loadfile = open("loadPyGeoloc", "w")  # write mode

            for y in np.append([8],np.arange(11, 16, 1)):
                for m in np.arange(1, 13, 1):
                    # print(('').join(
                    #     ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i)]))
                    loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rough_test), ' ', f'{y:02}', f'{m:02}', ' 1 ', str(i), '\n']))

            loadfile.close()

            loadfile = open("loadPyXover", "w")  # write mode

            for ymc in np.arange(0, 21, 1):
                    # print(('').join(
                    #     ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i)]))
                    loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rough_test), ' ', str(ymc), ' 2 ', str(i), '\n']))

            loadfile.close()

            print("Processing PyXover series at external iteration",i)
            iostat = 0
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadPyGeoloc', 'PyGeoloc', '8', '2:30:00', '10'])
            if iostat != 0:
                print("*** PyGeoloc failed on iter", i)
                exit(iostat)
            iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadPyXover', 'PyXover', '8', '99:99:99', '10'])
            if iostat != 0:
                print("*** PyXover failed on iter", i)
                exit(iostat)
            #iostat = s.call(['/home/sberton2/launchLISTslurm', 'loadAccSol', 'PyAccum', '8', '99:99:99', '10'])
            if iostat != 0:
                print("*** PyAccum failed on iter", i)
                exit(iostat)

            # stop clock and print runtime
            # -----------------------------
            end = time.time()
            print('----- Runtime tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
