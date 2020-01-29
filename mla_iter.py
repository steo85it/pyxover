import subprocess as s
import time

import numpy as np

from prOpt import local, sim_altdata

if __name__ == '__main__':

    # set up res and amp to 'loop onto' in launch_test.py 
    #(they should have been previously created by PyAltSim)
    rough_test = np.array([0]) #np.arange(1,6,1)

    for rt in rough_test:
        for i in np.arange(0, 2):
    
            if local:
                start = time.time()
                print("Processing PyXover series at external iteration",i)
                for y in np.arange(11, 16, 1): #np.append([8],np.arange(11, 16, 1)):
                    for m in np.arange(1, 13, 1):
                        # print(["python3", "launch_test.py", str(rough_test), ' ', str(y), f'{m:02}', "1", str(i)])
                        # exit()
                        ym = f'{y:02}'+ f'{m:02}'
                        iostat = s.call(["python3", "launch_test.py", str(rt), ym, "1", str(i)])
                        if iostat != 0:
                            print("*** PyGeoloc failed on iter", i)
                            exit(iostat)
                # stop clock and print runtime
                # -----------------------------
                end = time.time()

                print('----- Runtime PyGeoloc tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
                start = time.time()
    
                for ymc in np.arange(0, 21, 1):
                    iostat = s.call(["python3", "launch_test.py", str(rt), str(ymc), "2", str(i)])
                    if iostat != 0:
                        print("*** PyXover failed on iter", i)
                        # exit(iostat)
                # stop clock and print runtime
                # -----------------------------
                end = time.time()

                print('----- Runtime PyXover tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
                start = time.time()
    
                iostat = s.call(["python3", "launch_test.py", str(rt), "0", "3", str(i)])
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
                            ['python3 launch_test.py ', str(rt), ' ', f'{y:02}', f'{m:02}', ' 1 ', str(i), '\n']))

                loadfile.close()

                loadfile = open("loadPyXover", "w")  # write mode

                # prepend xov_analysis for years taking longer (to optimize cluster use)
                tmp = np.arange(0, 21, 1)
                xovlist = [16, 15, 11, 18, 17]
                for el in tmp:
                    if el not in xovlist:
                        xovlist.append(el)
                for ymc in np.array(xovlist):
                # for ymc in np.arange(0, 21, 1):
                    # print(('').join(
                    #     ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i)]))
                    loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rt), ' ', str(ymc), ' 2 ', str(i), '\n']))

                loadfile.close()

                loadfile = open("loadAccSol", "w")  # write mode

                loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rt), ' 0 3 ', str(i), '\n']))

                loadfile.close()

                print("Processing PyXover series at external iteration", i)
                iostat = 0
                iostat = s.call(
                    ['/home/sberton2/launchLISTslurm', 'loadPyGeoloc', 'PyGeo_' + str(rt)+'_'+str(i), '8', '00:30:00', '10'])
                if iostat != 0:
                    print("*** PyGeol_" + str(rt) + " failed on iter", i)
                    exit(iostat)
                iostat = s.call(
                    ['/home/sberton2/launchLISTslurm', 'loadPyXover', 'PyXov_' + str(rt)+'_'+str(i), '8', '00:45:00', '10'])
                if iostat != 0:
                    print("*** PyXov_" + str(rt) + " failed on iter", i)
                    exit(iostat)
                iostat = s.call(#["python3", "launch_test.py", str(rt), "0", "3", str(i)])
                    ['/home/sberton2/launchLISTslurm', 'loadAccSol', 'PyAcc_' + str(rt)+'_'+str(i), '1', '00:20:00', '1'])
                if iostat != 0:
                    print("*** PyAcc_" + str(rt) + " failed on iter", i)
                    exit(iostat)

                # stop clock and print runtime
                # -----------------------------
                end = time.time()
                print(
                    '----- Runtime tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
