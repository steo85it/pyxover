import glob
import logging
import os
# import numpy as np
import pandas as pd
import subprocess as s

from config import XovOpt
from examples.LOLA.prepro_LOLA import prepro_LOLA
from examples.LOLA_xov.process_rdr import process_rdr
from pygeoloc import PyGeoloc
from examples.LOLA_xov.setup_lola import setup_lola
from scipy.constants import speed_of_light as clight

# add to PYTHONPATH before running (should work with setup.py, but it doesn't...)
# export PYTHONPATH=/att/nobackup/sberton2/LOLA/PyXover/src:/att/nobackup/sberton2/LOLA/PyXover:$PYTHONPATH

def launch_slurm(filnamout,phase):

    print(filnamout)
    iostat = 0
    iostat = s.call(['/home/sberton2/launchLISTslurm',filnamout,'PyAltSim_'+str(phase),'1', '99:00:00', '20Gb', '100'])
    if iostat != 0:
        print("*** PyAltSimLOLA (phase "+str(phase)+") failed")
        exit(iostat)


if __name__ == '__main__':

    setup_lola()

    tmpdir = XovOpt.get("tmpdir") #'./' # '/home/sberton2/tmp/'
    filnamin = f'testLOLA.in'
    filnamout = f'loadPyAltSim'

    bin_rdrs = glob.glob(f"{XovOpt.get('rawdir')}/LOLARDR_*.DAT")
    if len(bin_rdrs) == 0:
        logging.error(f"No files found following the path `{XovOpt.get('rawdir')}/LOLARDR_*.DAT`")

    if True:
        for f in bin_rdrs[:10]:
            f = f.split('/')[-1].split('.')[0]
            df=process_rdr(f"{XovOpt.get('rawdir')}",f)
            # pd.set_option('max_columns', None)
            # print(df.head())
            # print(df.columns)

            df['met']=df[['met_seconds']].values + df[['subseconds']].values
            df.reset_index(inplace=True)

            # select only data from RoI
            df = df.loc[df["latitude_1"]<-87.].reset_index(drop=True)

            # dflist=[]
            print(f"Converting {f} to ascii...")

            for i in range(6)[1:]:
                tmp=df.rename(columns={f"longitude_{i}":"geoc_long",f"latitude_{i}":"geoc_lat",f"radius_{i}":"altitude",
                                       "transmit_time":"EphemerisTime","met":"MET",
                                   f"shot_flag_{i}":"chn","threshold_1":"thrsh",f"gain_{i}":"gain",f"range_{i}":"TOF_ns_ET",
                                   "sc_longitude":"Sat_long","sc_latitude":"Sat_lat","sc_radius":"Sat_alt","offnadir_angle":"Offnad",
                                   "solar_phase":"Phase","solar_incidence":"Sol_inc","index":"seqid"},errors='raise')
                tmp[['frm', 'Pulswd', '1way_range', 'Emiss', 'TXmJ', 'UTC', 'SCRNGE']] = None
                # tmp[["TOF_ns_ET"]]/=clight

                mla_cols = ['rdr_name','geoc_long','geoc_lat','altitude','EphemerisTime','MET','frm','chn','Pulswd','thrsh','gain','1way_range','Emiss','TXmJ','UTC','TOF_ns_ET','Sat_long','Sat_lat','Sat_alt','Offnad','Phase','Sol_inc','SCRNGE','seqid']

                rdr_year = tmp.rdr_name.values[0].split('_')[-1][:2]
                prepro_outdir = f"{XovOpt.get('rawdir')}SIM_{rdr_year}/{XovOpt.get('expopt')}/0res_{i}amp/"
                os.makedirs(prepro_outdir, exist_ok=True)
                tmp[mla_cols].to_pickle(f"{prepro_outdir}{f}.pkl")

    # processing
    print("Processing started...")
    f = open(filnamout,'w')
    f.truncate(0)
    # write file
    for rdr_id in bin_rdrs:
        rdr_id = rdr_id.split('/')[-1].split('.')[0].split('_')[-1]
        print(rdr_id)
        os.makedirs(f"{XovOpt.get('auxdir')}{rdr_id}/slewcheck_0", exist_ok=True)
        print("0. 0. 0.", file=open(f"{XovOpt.get('auxdir')}{rdr_id}/slewcheck_0/_boresights_LOLA_ch12345_night_laser2_fov_bs0.inc", 'w'))
        f.write((' ').join([f'python3 lola_interface.py 0 {rdr_id} 1', '\n']))
    f.close()
    exit()
    # TODO use the more advanced launch_slurm with local option
    launch_slurm(filnamout,phase=1)

    # postpro
    print("Post-processing started...")
    postpro_cols = ['trk']
    strs = df[prepro_cols].values
    f = open(filnamout,'w')
    f.truncate(0)
    # write file
    for row in range(strs.shape[0]):
        f.write((' ').join(['python3 postpro_LOLA.py', str(strs[row,0]), '\n']))
    f.close()
    launch_slurm(filnamout,phase=2)
