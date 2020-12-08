# import shutil
import subprocess as s
import time

import numpy as np

from examples.MLA.options import local, sim_altdata, monthly_sets, outdir, datasimopt, resopt, amplopt

if __name__ == '__main__':

    # set up res and amp to 'loop onto' in launch_test.py 
    #(they should have been previously created by PyAltSim)
    rough_test = np.array([0]) #np.arange(1,6,1)

    for rt in rough_test:
        for iter in np.arange(0,1):
    
            if local:
                print("Processing PyXover series at external iteration", iter)

                # Preliminary step to fit orbits and pointing to current knowledge of topography (direct altimetry)
                if iter == 0 and False:
                    start = time.time()
                    for y in [12,13]: #np.append([8],np.arange(11, 16, 1)):
                        for m in [1]: # np.arange(1, 13, 1):
                            # print(["python3", "launch_test.py", str(rough_test), ' ', str(y), f'{m:02}', "1", str(i)])
                            # exit()
                            ym = f'{y:02}' + f'{m:02}'

                            iostat = s.call(["python3", "launch_test.py", str(rt), ym, "1", str(iter)])
                            if iostat != 0:
                                print("*** PyGeoloc failed on iter", iter)
                                # exit(iostat)

                            iostat = s.call(["python3", "fit2dem.py", ym])
                            if iostat != 0:
                                print("*** fit2dem failed on iter", iter)
                                exit(iostat)
                    # stop clock and print runtime
                    # -----------------------------
                    end = time.time()

                    print('----- Runtime fit2dem tot = ' + str(end - start) + ' sec -----' + str(
                        (end - start) / 60.) + ' min -----')

                start = time.time()
                for y in [12,13]: #np.append([8],np.arange(11, 16, 1)):
                    for m in [1]: #np.arange(1, 13, 1):
                        # print(["python3", "launch_test.py", str(rough_test), ' ', str(y), f'{m:02}', "1", str(i)])
                        # exit()
                        ym = f'{y:02}'+ f'{m:02}'
                        iostat = s.call(["python3", "launch_test.py", str(rt), ym, "1", str(iter)])
                        if iostat != 0:
                            print("*** PyGeoloc failed on iter", iter)
                            exit(iostat)
                # stop clock and print runtime
                # -----------------------------
                end = time.time()

                print('----- Runtime PyGeoloc tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')

                start = time.time()
    
                for ymc in [12]: #np.arange(0, 21, 1):
                    iostat = s.call(["python3", "launch_test.py", str(rt), str(ymc), "2", str(iter)])
                    if iostat != 0:
                        print("*** PyXover failed on iter", iter)
                        # exit(iostat)
                # stop clock and print runtime
                # -----------------------------
                end = time.time()

                print('----- Runtime PyXover tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
                start = time.time()
    
                iostat = s.call(["python3", "launch_test.py", str(rt), "0", "3", str(iter)])
                print("iostat", iostat)
                if iostat != 0:
                    print("*** PyAccum failed on iter", iter)
                else:
                    out_path = outdir+'/'+datasimopt+'/'+str(resopt[0])+'res_'+str(amplopt[0])+'amp/'
                    from pathlib import Path

                    print(out_path)
                    print(Path(out_path))
                    for p in Path(out_path).glob('gtrack_*'):
                        p.unlink()
                        print(p,"has been succesfully removed after solution!")

                # stop clock and print runtime
                # -----------------------------
                end = time.time()
                print('----- Runtime AccumXov tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
    
            else:
                start = time.time()
                loadfile = open("loadPyGeoloc", "w")  # write mode
                load_fit2dem = open("loadfit2dem", "w")  # write mode

                for y in [12]: #np.append([8],np.arange(11, 16, 1)):

                    for m in np.arange(1,2,1): #(1, 13, 1):
                        # print(('').join(
                        #     ['python3 launch_test.py ', str(rough_test), ' ', str(y), f'{m:02}', ' 1 ', str(i)]))
                        loadfile.write(('').join(
                            ['python3 launch_test.py ', str(rt), ' ', f'{y:02}', f'{m:02}', ' 1 ', str(iter), '\n']))
                        load_fit2dem.write(('').join(
                            ['python3 fit2dem.py ', f'{y:02}', f'{m:02}', '\n']))

                loadfile.close()
                load_fit2dem.close()

                if monthly_sets:
                 for i in range(2):
                   loadfile = open("loadPyXover_"+str(i), "w")  # write mode
                   xovlist = np.arange(i*689,(i+1)*689,1)

                else:
                   loadfile = open("loadPyXover", "w")  # write mode
                   # prepend xov_analysis for years taking longer (to optimize cluster use)
                   tmp = np.arange(12,13,1) #(0, 21, 1)
                   xovlist = [] #[16, 12, 13, 8, 15, 7, 9, 11, 17, 18, 14, 19, 10, 6, 20]
                   for el in tmp:
                          if el not in xovlist:
                            xovlist.append(el)

                for ymc in np.array(xovlist):
                       loadfile.write(('').join(
                           ['python3 launch_test.py ', str(rt), ' ', str(ymc), ' 2 ', str(iter), '\n']))

                loadfile.close()

                loadfile = open("loadAccSol", "w")  # write mode

                loadfile.write(('').join(
                        ['python3 launch_test.py ', str(rt), ' 0 3 ', str(iter), '\n']))

                loadfile.close()

                print("Processing PyXover series at external iteration", iter)

                iostat = 0

                # Preliminary step to fit orbits and pointing to current knowledge of topography (direct altimetry)
                # (real data only)
                if iter == 0 and False:
                    iostat = s.call(
                        ['/home/sberton2/launchLISTslurm', 'loadPyGeoloc', 'PyGeo_' + str(rt) + '_' + str(-1), '8',
                         '01:30:00', '90Gb', '10'])
                    if iostat != 0:
                        print("*** PyGeol_" + str(rt) + " failed on iter", str(-1))
                        exit(iostat)
                    iostat = s.call(
                        ['/home/sberton2/launchLISTslurm', 'loadfit2dem', 'fit2dem_' + str(rt) + '_' + str(-1), '8',
                         '03:00:00', '90Gb', '10'])
                    if iostat != 0:
                        print("*** fit2dem_" + str(rt) + " failed on iter", str(-1))
                        exit(iostat)

                if iter > 1:
                    iostat = s.call(
                        ['/home/sberton2/launchLISTslurm', 'loadPyGeoloc', 'PyGeo_' + str(rt) +'_' + str(iter), '7', '01:00:00', '30Gb', '10'])
                    if iostat != 0:
                        print("*** PyGeol_" + str(rt) + " failed on iter", iter)
                        exit(iostat)

                if monthly_sets:
                  for i in range(2):
                    iostat = s.call(
                        ['/home/sberton2/launchLISTslurm', 'loadPyXover_'+str(i), 'PyXov_' + str(rt) +'_' + str(iter)+'_' + str(i), '1', '06:00:00', '5Gb', '100'])
                    if iostat != 0:
                        print("*** PyXov_" + str(rt) + " failed on iter", iter)
                        exit(iostat)
                else:
                  iostat = s.call(
                      ['/home/sberton2/launchLISTslurm', 'loadPyXover', 'PyXov_' + str(rt) +'_' + str(iter), '8', '04:00:00', '99Gb', '10'])
                  if iostat != 0:
                      print("*** PyXov_" + str(rt) + " failed on iter", iter)
                      exit(iostat)


                iostat = s.call(#["python3", "launch_test.py", str(rt), "0", "3", str(i)])
                    ['/home/sberton2/launchLISTslurm', 'loadAccSol', 'PyAcc_' + str(rt) +'_' + str(iter), '8', '02:30:00', '90Gb', '1'])
                if iostat != 0:
                    print("*** PyAcc_" + str(rt) + " failed on iter", iter)
                    exit(iostat)

                # stop clock and print runtime
                # -----------------------------
                end = time.time()
                print(
                    '----- Runtime tot = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
