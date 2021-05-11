import logging
import os
import unittest

import pandas as pd
from numpy.testing import assert_array_equal

from accumxov.accum_opt import AccOpt
from config import XovOpt
from examples.lidar_moon.swath_h2 import main as swath_main


class LidarXoverTest(unittest.TestCase):

    def setUp(self) -> None:
        # update paths and check options
        XovOpt.set("body", 'MOON')  #
        XovOpt.set("instrument", "pawstel")  # needs to be set directly in config, too, else error (accum_opt doesn't see the updated value)
        XovOpt.set("basedir", 'lidar_moon/data/')
        # vite fait...
        vecopts = {'SCID': '-236',
                   'SCNAME': 'PAWSTEL',
                   'SCFRAME': -236000,
                   'INSTID': (-236500, -236501),
                   'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
                   'PLANETID': '199',
                   'PLANETNAME': 'MOON',
                   'PLANETRADIUS': 2440.,
                   'PLANETFRAME': 'IAU_MERCURY',
                   'OUTPUTTYPE': 1,
                   'ALTIM_BORESIGHT': '',
                   'INERTIALFRAME': 'J2000',
                   'INERTIALCENTER': 'SSB',
                   'PM_ORIGIN': 'J2013.0',
                   'PARTDER': ''}
        XovOpt.set('vecopts',vecopts)

        XovOpt.set("sol4_orb", [])
        XovOpt.set("sol4_glo", ['dR/dh2'])

        XovOpt.set("par_constr", {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 5.e0, 'dR/dA': 1.e2,
                      'dR/dC': 1.e2, 'dR/dR': 5.e-3,  # } #, 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
                      'dR/dA1': 1.e-1, 'dR/dC1': 1.e-1, 'dR/dR1': 1.e2, 'dR/dA2': 1.e-2, 'dR/dC2': 1.e-2, 'dR/dR2': 1.e2,
                      'dR/dAC': 1.e-1, 'dR/dCC': 1.e-1, 'dR/dRC': 1.e2, 'dR/dAS': 1.e-2, 'dR/dCS': 1.e-2,
                      'dR/dRS': 1.e2})  # , 'dR/dA2':1.e-4, 'dR/dC2':1.e-4,'dR/dR2':1.e-2} # 'dR/dA':100., 'dR/dC':100.,'dR/dR':100.} #, 'dR/dh2': 1} #
        XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 5.e-2})

        AccOpt.check_consistency()

    def test_swath_pipeline_cnt(self):

        # test with dR + dh2 solution
        XovOpt.set("OrbRep", 'cnt')
        XovOpt.set("sol4_orbpar", ['dR']) # None] #['dA','dC', #,'dRl','dPt'] #] #
        XovOpt.check_consistency()

        print(XovOpt.get("basedir"))
        # exit()
        df_results, df_xovers = swath_main()
        # print(df_results)
        # print(df_xovers)
        refdir = f'{XovOpt.get("basedir")}../reference/24H/{XovOpt.get("OrbRep")}/'
        ref_results = pd.read_csv(f'{refdir}df_results_dlon30.csv', index_col=0)
        ref_xovers = pd.read_csv(f'{refdir}df_xovers_dlon30.csv', index_col=0)

        assert_array_equal(df_xovers.values.round(3), ref_xovers.values.round(3))
        assert_array_equal(df_results.values.round(3), ref_results.values.round(3))

    def test_swath_pipeline_per(self):

        # test with periodic orb + dh2 solution
        XovOpt.set("OrbRep", 'per')
        XovOpt.set("sol4_orbpar", ['dRC','dRS']) # None] #['dA','dC', #,'dRl','dPt'] #] #
        XovOpt.check_consistency()

        print(XovOpt.get("basedir"))
        # exit()

        df_results, df_xovers = swath_main()
        # print(df_results)
        # print(df_xovers)
        refdir = f'{XovOpt.get("basedir")}../reference/24H/{XovOpt.get("OrbRep")}/'
        ref_results = pd.read_csv(f'{refdir}df_results_dlon30.csv', index_col=0)
        ref_xovers = pd.read_csv(f'{refdir}df_xovers_dlon30.csv', index_col=0)

        assert_array_equal(df_xovers.values.round(3), ref_xovers.values.round(3))
        assert_array_equal(df_results.values.round(3), ref_results.values.round(3))


    def TearDown(self):
        logging.info("TestLidarXover done!")