import os
# import numpy as np
#import pandas as pd
#import subprocess as s

from config import XovOpt
#from prepro_LOLA import prepro_LOLA
#from pygeoloc import PyGeoloc

def setup_lola():

    XovOpt.set("instrument", 'LOLA')
    XovOpt.set("expopt", 'LX0')
    XovOpt.set("body", 'MOON')
    XovOpt.set("local", True)
    XovOpt.set("partials", False)
    XovOpt.set("new_sim", 2)
    XovOpt.set("apply_topo", True)
    XovOpt.set("SpInterp", 0)
    XovOpt.set("local_dem", True)
    
    vecopts = {'SCID': '-85',
               'SCNAME': 'LRO',
               'SCFRAME': 'LRO_SC_BUS', # '-85000',
               'INSTID': (0, 0),
               'INSTNAME': ('', ''),
               'PLANETID': '10',
               'PLANETNAME': 'MOON',
               'PLANETRADIUS': 1737.4,
               'PLANETFRAME': 'MOON_ME',
               'OUTPUTTYPE': 1,
               'ALTIM_BORESIGHT': '',
               'INERTIALFRAME': 'J2000',
               'INERTIALCENTER': 'SSB',
               'PARTDER': ''}
    XovOpt.set("vecopts",vecopts)

    XovOpt.set("basedir","data/") #"/att/nobackup/sberton2/LOLA/PyXover/examples/LOLA/"
#    XovOpt.set("outdir",f'{basedir}out/')
#    XovOpt.set("auxdir",f'{basedir}aux/')
    XovOpt.set("inpdir",'data/') #/att/nobackup/dmao1/LOLA/slew_check/')
    # XovOpt.set("furnshdir",f"{XovOpt.get('basedir')}furnsh/")
    
    for newdir in ["tmpdir","outdir","auxdir","rawdir"]:
                os.makedirs(XovOpt.get("basedir")+XovOpt.get(newdir),exist_ok=True)

    XovOpt.check_consistency()
