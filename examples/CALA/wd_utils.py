#!/usr/bin/env python3

def build_sessiontable_man(MANFIL,darc_min,darc_max):
    import os
    import datetime as dt

    # Build session table
    # -------------------

    # Inputs
    #my $MANFIL = $_[0];
    #my $darc_min = $_[1]; # minimum and target duration
    #my $darc_max; # maximum duration
    # $darc_max = $_[2]; # maximum duration

    print("\nBuild session table\n")
    print("--------------------\n")

    if (not os.path.isfile(MANFIL)):
      print(MANFIL, "doesn't exist")
      exit()

    nSess = -1
    secsSinceJ2000 = []
    print(f"maneuver file {MANFIL}")
    with open(MANFIL,'r') as file:
       lines = file.readlines()
       for orbLine in lines:
        # Seconds since J2000 from $orbLine
        tokens = orbLine.split()
        secSinceJ2000 = int(tokens[0])
        if (secSinceJ2000 == ""):
          secSinceJ2000 = int(tokens[1])
        if (nSess==-1):
            nSess+=1
            secsSinceJ2000.append(secSinceJ2000)
        if (secsSinceJ2000[-1] != secSinceJ2000): # Arc without manveuvers
            print(f"\nArc no {nSess+1} from {secsSinceJ2000[-1]} to {secSinceJ2000} w/o maneuvers")
            if (darc_min != 0):
                duration = secSinceJ2000 - secsSinceJ2000[-1]
                duration_h = duration/3600

                print(f"Total arc duration = {duration_h} hours")

                number_arc = int(duration_h/darc_min - 0.5)
                if(darc_max == 0):
                    darc_max = int(duration_h/number_arc + 0.5)
                print(f"darc_mac is {darc_max}")
                duration_arc = darc_min
                # Looking for darc = k hours minimizing (narc*darc - duration)
                # (close to 24h but larger than 24h)
                for darc in range(darc_min,darc_max+1):
                    narc = int(duration_h/darc + 0.5)# ceil
                    if( abs(narc*darc - duration_h) < abs(number_arc*duration_arc - duration_h) ):
                        number_arc   = narc
                        duration_arc = darc
                print(f"{number_arc} arcs of {duration_arc} hours")

                for j in range(1,number_arc):
                    secsSinceJ2000.append(secsSinceJ2000[-1] + duration_arc*3600)
                    nSess+=1
            # Last arc is shorter
            print(f"Last arc lasts {(secSinceJ2000 - secsSinceJ2000[-1])/3600}h")
            nSess+=1
            secsSinceJ2000.append(secSinceJ2000)

    print(f"\nSession table built of {nSess-1} sessions for {(secsSinceJ2000[-1]-secsSinceJ2000[0])/86400} days")
    
    d_sess = [dt.datetime(2000,1,1,12,0,0) + dt.timedelta(seconds=sec) for sec in secsSinceJ2000]
    return d_sess

def xov_pkl2csv(pyout_folder, csv_filename, arc1, arc2, ids, track_lists):
    import numpy as np
    import pickle
    import csv
    import os.path
    import time

    file = open(csv_filename,'w')
    columns = ' tA int       |'\
            + ' tA frac               |'\
            + ' tB int        |'\
            + ' tB frac               |'\
            + ' Lat (deg)             |'\
            + ' Lon (deg)             |'\
            + ' dR                    |'\
            + ' distA                 |'\
            + ' distB                 |'\
            + ' dR/dR_A               |'\
            + ' dR/dA_A               |'\
            + ' dR/dC_A               |'\
            + ' dR/dR_B               |'\
            + ' dR/dA_B               |'\
            + ' dR/dC_B               |'\
            + ' dR/dRA                |'\
            + ' dR/dDEC               |'\
            + ' dR/dPM                |'\
            + ' dR/dL                 |'\
            + ' dR/dh2                |'
    file.write(f"#{columns}\n")
        
    writer = csv.writer(file, delimiter='\t')

    for id, track_list in zip(ids,track_lists):
      in_folder = f"{pyout_folder}{id}_0/3res_20amp/"

      xov_file_path = f"{in_folder}xov/xov_{arc1}_{arc2}.pkl"

      with open(xov_file_path, "rb") as xov_file:
         xov = pickle.load(xov_file)

      # columns=['x0', 'y0', 'mla_idA', 'mla_idB', 'cmb_idA', 'cmb_idB', 'R_A', 'R_B', 'dR'])
      tracks_listA = xov.xovers['orbA'].values
      tracks_listB = xov.xovers['orbB'].values

      if len(track_list) == 0:
         startInit = time.time()
         track_list = dict.fromkeys(np.append(tracks_listA,tracks_listB),0)

         for track_name in track_list:
            track_file_path = f"{in_folder}gtrack_{arc1}/gtrack_{track_name}.pkl"
            if (not os.path.isfile(track_file_path)):
               track_file_path = f"{in_folder}gtrack_{arc2}/gtrack_{track_name}.pkl"
            with open(track_file_path, "rb") as f:
               track = pickle.load(f)
               track_list[track_name] = track.t0_orb
         endInit = time.time()
         print(endInit-startInit)
      for i in range(0,len(tracks_listA)):
        t0_A = track_list[tracks_listA[i]]
        t0_B = track_list[tracks_listB[i]]
        if (i>1): # Check for duplicates
           if (xov.xovers['dtA'].values[i-1]  == xov.xovers['dtA'].values[i] and
               xov.xovers['dtB'].values[i-1]  == xov.xovers['dtB'].values[i]):
               continue
        tA_int = np.floor(t0_A + xov.xovers['dtA'].values[i])
        tB_int = np.floor(t0_B + xov.xovers['dtB'].values[i])

        row = [int(tA_int)]
        row.append((t0_A - tA_int) + xov.xovers['dtA'].values[i])
        row.append(int(tB_int))
        row.append((t0_B - tB_int) + xov.xovers['dtB'].values[i])
        row.append(xov.xovers['LAT'].values[i])
        row.append(xov.xovers['LON'].values[i])
        row.append(xov.xovers['dR'].values[i])
        row.append(min(xov.xovers['dist_Am'].values[i],xov.xovers['dist_Ap'].values[i]))
        row.append(min(xov.xovers['dist_Bm'].values[i],xov.xovers['dist_Bp'].values[i]))
        row.append(xov.xovers['dR/dR_A'].values[i])
        row.append(xov.xovers['dR/dA_A'].values[i])
        row.append(xov.xovers['dR/dC_A'].values[i])
        row.append(xov.xovers['dR/dR_B'].values[i])
        row.append(xov.xovers['dR/dA_B'].values[i])
        row.append(xov.xovers['dR/dC_B'].values[i])
        row.append(xov.xovers['dR/dRA'].values[i])
        row.append(xov.xovers['dR/dDEC'].values[i])
        row.append(xov.xovers['dR/dPM'].values[i])
        row.append(xov.xovers['dR/dL'].values[i])
        row.append(xov.xovers['dR/dh2'].values[i])
        if np.isnan(np.sum(row)):
           print("Found nan in")
           print(row)
           continue

        writer.writerow(row)

      endInit = time.time()
    file.close()

def getgtrackt0(pyout_folder, id):
    import pickle
    import os.path

    in_folder = f"{pyout_folder}{id}_0/3res_20amp/"
    list_dir = os.listdir(in_folder)
    list_arcs = [folder.split("_")[-1] for folder in list_dir if ((len(folder) == 13) & (folder.split("_")[0]=='gtrack'))]

    track_list = dict()
    for arc1 in list_arcs:
      list_file = os.listdir(f'{in_folder}gtrack_{arc1}/')
      track_list0 = [file.split(".")[-2].split('_')[1] for file in list_file if file.split(".")[-1]=='pkl']
      track_list.update(dict.fromkeys(track_list0))
      for track_name in track_list0:
         track_file_path = f"{in_folder}gtrack_{arc1}/gtrack_{track_name}.pkl"
         with open(track_file_path, "rb") as f:
            track = pickle.load(f)
            track_list[track_name] = track.t0_orb
    return track_list