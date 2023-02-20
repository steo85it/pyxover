# Options configuration for pyxover applications
import multiprocessing as mp
import numpy as np
from xovutil.units import deg2as

class XovOpt:

    __conf = {
        # env opt
        "debug": False,
        "local": True, #
        "parallel": False,
        "partials": True,
        "unittest": False,

        "body" : 'MERCURY', #
        "instrument" : "MLA", #"pawstel", #"BELA", #
        "selected_hemisphere" : 'N',

        # dirs
        "basedir": 'pawstel/data/',
        "rawdir": f'raw/',
        "outdir": f'out/',
        "auxdir": f'aux/',
        "tmpdir": f'tmp/',
        "inpdir": f'',
        "spauxdir": 'KX_spk/',  # 'AG_AC_spk/' #'KX_spk/' #'OD380_spk/' #'AG_spk/'

        # pyxover options
        # set number of processors to use
        "n_proc": mp.cpu_count() - 3,
        "import_proj": False,
        "import_abmat": "",

        # processing opt
        "expopt" : 'BS0',
        "resopt" : [0],
        "amplopt" : [1],

        "parOrb": {'dA': 20., 'dC': 20., 'dR': 5.},  # ,'dRl':0.2, 'dPt':0.2} #
        "parGlo": {'dRA': [0.2, 0.000, 0.000], 'dDEC': [0.36, 0.000, 0.000], 'dPM': [0, 0.013, 0.000],
              'dL': 1.e-3 * deg2as(1.) * np.linalg.norm([0.00993822, -0.00104581, -0.00010280, -0.00002364, -0.00000532]), 'dh2': 0.1},

        # parameter constraints for solution
        "par_constr": {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 3.e-1, 'dR/dA': 1.e2,
                  'dR/dC': 1.e2, 'dR/dR': 2.e1},  # , 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
        # 'dR/dA1':1.e-1, 'dR/dC1':1.e-1,'dR/dR1':1.e-1, 'dR/dA2':1.e-2, 'dR/dC2':1.e-2,'dR/dR2':1.e-2} #, 'dR/dA2':1.e-4, 'dR/dC2':1.e-4,'dR/dR2':1.e-2} # 'dR/dA':100., 'dR/dC':100.,'dR/dR':100.} #, 'dR/dh2': 1} #
        "mean_constr": {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 1.e0},  # , 'dR/dRl':1.e-1, 'dR/dPt':1.e-1}
        # define if it's a closed loop simulation run
        "cloop_sim": False,

        # perturbations for closed loop sims (dRl, dPt, dRA, dDEC, dL in arcsec; dPM in arcsec/Julian year)
        "pert_cloop_orb": {},  # 'dA':50., 'dC':50., 'dR':20.,'dRl':0.5, 'dPt':0.5} #} #, 'dA1':20., 'dC1':20., 'dR1':5.
        # in deg and deg/day as reminder pert_cloop_glo": {'dRA':[0.0015deg, 0.000, 0.000], 'dDEC':[0.0015deg, 0.000, 0.000],'dPM':[0, 2.e-6deg/day, 0.000],'dL':~3*1.5as, 'dh2':-1.} # compatible with current uncertitudes
        "pert_cloop_glo": {},  # 'dRA':[3.*5., 0.000, 0.000], 'dDEC':[3.*5., 0.000, 0.000],'dPM':[0, 3.*3., 0.000],'dL':3.*deg2as(1.5*0.03)*np.linalg.norm([0.00993822,-0.00104581,-0.00010280,-0.00002364,-0.00000532]), 'dh2':-1.} #
        "pert_cloop": {}, # initialized by check_consistency below
        # perturb individual tracks
        "pert_tracks": [],  # '1107021838','1210192326','1403281002','1503191143'] #

        # select subset of parameters to solve for
        "sol4_orb": [None],  # '1503250029'] #'1107021838','1210192326','1403281002','1503191143']  #
        "sol4_orbpar": [None],  # ['dA','dC','dR'] #,'dRl','dPt'] #,'dA1','dC1','dR1','dA2','dC2','dR2']  #] #
        "sol4_glo": ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL'],  # ,'dR/dh2'] #,  None]

        # orbital representation
        "OrbRep": 'cnt' , # 'lin' # 'quad' #

        # interpolation/spice direct call (0: use spice, 1: yes, use interpolation, 2: yes, create interpolation)
        "SpInterp": 0,
        # create new gtrack (0:no, 1:yes, if not already present, 2: yes, create and replace)
        "new_gtrack": 2,
        # create new xov (0:no, 1:yes, if not already present, 2: yes, create and replace)
        "new_xov": 2,

        # Other options
        # monthly or yearly sets for PyXover
        "monthly_sets": False,
        # analyze multi-xov pairs
        "multi_xov": False,
        # compute full covariance (could give memory issues)
        "full_covar": False,  # True #
        # roughness map
        "roughn_map": False,
        # new algo
        "new_algo": True,  # False #
        # load input xov
        "compute_input_xov": True,

        # PyAltSim options
        # simulation mode ! WD: possible option to (0:no, 1:yes, use, 2: yes, create)
        "sim_altdata": 0,
        # recompute a priori
        "new_illumNG": 0,
        # interpolation/spice direct call (0:no, 1:yes, use, 2: yes, create)
        "new_sim" : 2,
        # use large scale topography
        "apply_topo": 0,
        # apply small scale topography (simulated)
        "small_scale_topo": False,
        # range noise
        "range_noise": 0,
        # local/global DEM (LOLA)
        "local_dem": True,
        
        # vecopts
        # Setup some useful options
        "vecopts": {'SCID': '-236',
                   'SCNAME': 'MESSENGER',
                   'SCFRAME': -236000,
                   'INSTID': (-236500, -236501),
                   'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
                   'PLANETID': '199',
                   'PLANETNAME': 'MERCURY',
                   'PLANETRADIUS': 2440.,
                   'PLANETFRAME': 'IAU_MERCURY',
                   'OUTPUTTYPE': 1,
                   'ALTIM_BORESIGHT': '',
                   'INERTIALFRAME': 'J2000',
                   'INERTIALCENTER': 'SSB',
                   'PM_ORIGIN': 'J2013.0',
                   'PARTDER': ''}
    }
    __setters = list(__conf.keys())

    @staticmethod
    def check_consistency():

        XovOpt.set("rawdir", f'{XovOpt.get("basedir")}raw/'),
        XovOpt.set("outdir", f'{XovOpt.get("basedir")}out/'),
        XovOpt.set("auxdir", f'{XovOpt.get("basedir")}aux/'),
        XovOpt.set("tmpdir", f'{XovOpt.get("basedir")}tmp/'),

        XovOpt.set("import_abmat",
                   (False, XovOpt.get("outdir") + "sim/BS2_0/0res_1amp/Abmat*.pkl")),
        XovOpt.set("pert_cloop", {'orb': XovOpt.get("pert_cloop_orb"),
                                  'glo': XovOpt.get("pert_cloop_glo")}),

        # if get doesn't work, use this
        if XovOpt.__conf['body'] != XovOpt.get("vecopts")['PLANETNAME']:
            raise NameError(f"Body name {XovOpt.__conf['body']}"
                            f" is inconsistent with vecopts attr "
                            f"{XovOpt.get('vecopts')['PLANETNAME']}."
                            f" Please update vecopts via XovOpt.set.")

    @staticmethod
    def get(name):
        return XovOpt.__conf[name]

    @staticmethod
    def set(name, value):
        if name in XovOpt.__setters:
            XovOpt.__conf[name] = value
            print(f"### XovOpt.{name} updated to {value}.")
        else:
            raise NameError("Name not accepted in XovOpt.set() method")

    @staticmethod
    def display():
        print(XovOpt.__conf)

    @staticmethod
    def to_dict():
        return XovOpt.__conf

    @staticmethod
    def clone(opts):
        # print("- Updating XovOpt")
        XovOpt.__conf = opts.copy()
        
# example, suppose importing
# from config import Options
if __name__ == '__main__':

    print(XovOpt.get("vecopts"))

    opt = XovOpt()
    print(opt.get("vecopts"))

    print(opt.get("body"))
    opt.set("body","MOON") # this causes an issue
    print(opt.get("body"))

    print(opt.get("tmpdir"))
    opt.set("basedir",'/att/nobackup/sberton2/MLA/')

    opt.check_consistency()
    print(opt.get("tmpdir"))
