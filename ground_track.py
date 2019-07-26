#!/usr/bin/env python3
# ----------------------------------
# ground_track.py
#
# Description: 
# 
# ----------------------------------------------------
# Author: Stefano Bertone
# Created: 08-Feb-2019
import os
import pickle
import time

import numpy as np
import pandas as pd
import spiceypy as spice

import astro_trans as astr
import pickleIO
from geolocate_altimetry import geoloc
from interp_obj import interp_obj
from prOpt import debug, partials, parallel, SpInterp, auxdir, parOrb, parGlo, pert_cloop, pert_tracks, sim_altdata, local
# from mapcount import mapcount
from project_coord import project_stereographic
from tidal_deform import tidepart_h2
from util import mergsum


class gtrack:
    interp_obj.interp = interp_obj.interpCby  # Cby  # Spl #
    interp_obj.eval = interp_obj.evalCby  # Cby  # Spl #

    def __init__(self, vecopts):

        self.vecopts = vecopts
        self.dr_simit = None
        self.ladata_df = None
        self.df_input = None
        self.name = None
        self.MERv = None
        self.MERx = None
        self.MGRa = None
        self.MGRv = None
        self.MGRx = None
        self.param = None
        # Set-up empty offset arrays at init (sim only)
        self.pertPar = {'dA': 0.,
                        'dC': 0.,
                        'dR': 0.,
                        'dRl': 0.,
                        'dPt': 0.,
                        'dRA': [0., 0., 0.],
                        'dDEC': [0., 0., 0.],
                        'dPM': [0., 0., 0.],
                        'dL': 0.}  # ,
        # 'dh2': 0. }
        self.pert_cloop = None
        self.sol_prev_iter = None
        self.t0_orb = None

    # create groundtrack from data and save to file
    def setup(self, filnam):

        # read data and fill ladata_df
        # self.read_fill(filnam)

        if len(self.ladata_df) > 0:
            if not hasattr(self, 'SpObj') and SpInterp == 2:
                # print(filnam)
                # print(self.ladata_df)
                # create interp for track
                self.interpolate()
            else:
                self.SpObj = pickleIO.load(auxdir + 'spaux_' + self.name + '.pkl')

            if debug:
                pd.set_option('display.max_columns', 500)

            # geolocate observations in orbit
            self.geoloc()
            # project observations
            self.project()
            # update df

            # print(self.ladata_df)

    # create groundtrack from data and save to file
    def prepro(self, filnam):

        # read data and fill ladata_df
        self.read_fill(filnam)
        # print(self.ladata_df)

        # testInterp(self.ladata_df,self.vecopts)
        # exit()

        # check spk coverage and select obs in ladata_df
        # self.check_coverage()

        # np.sum([t[0] <= val <= t[1] for t in twind])
        # print(self.ladata_df.loc[np.sum([t[0] <= self.ladata_df.ET_TX <= t[1] for t in twind])])
        # exit()

        # create interp for track (if data are present)
        if (len(self.ladata_df) > 0):
            if not hasattr(self, 'SpObj') and SpInterp == 2:
                # print(filnam)
                # print(self.ladata_df)
                # create interp for track
                self.interpolate()
            else:
                self.SpObj = pickleIO.load(auxdir + 'spaux_' + self.name + '.pkl')
        else:
            print('No data selected for orbit ' + str(self.name))
        # print(self.MGRx.tck)

    def check_coverage(self):
        cover = spice.utils.support_types.SPICEDOUBLE_CELL(2000)
        if local == 0:
             spice.spkcov(auxdir+'spk/MSGR_HGM008_INTGCB.bsp', -236, cover)
        else:
             spice.spkcov('/home/sberton2/Works/NASA/Mercury_tides/spktst/MSGR_HGM008_INTGCB.bsp', -236, cover)
        
        twind = [spice.wnfetd(cover, i) for i in range(spice.wncard(cover))]
        epo_in = np.sort(self.ladata_df.ET_TX.values)
        self.ladata_df['in_spk'] = np.array([np.sum([t[0] <= val <= t[1] for t in twind]) for val in epo_in]) > 0
        if debug:
            print(len(self.ladata_df.loc[self.ladata_df['in_spk'] == False]))
            print("lensel", len(self.ladata_df), len(self.ladata_df.loc[self.ladata_df['in_spk']]))
        self.ladata_df = self.ladata_df.loc[self.ladata_df['in_spk']]

    # create groundtrack from list of epochs
    def simulate(self, filnam):

        # read data and fill ladata_df
        self.read_fill(filnam)
        # print(self.ladata_df)

        # testInterp(self.ladata_df,self.vecopts)
        # exit()

        # create interp for track (if data are present)
        if (len(self.ladata_df) > 0):
            if not hasattr(self, 'SpObj') and SpInterp == 2:
                # print(filnam)
                # print(self.ladata_df)
                # create interp for track
                self.interpolate()
            else:
                self.SpObj = pickleIO.load(auxdir + 'spaux_' + self.name + '.pkl')
        else:
            print('No data selected for orbit ' + str(self.name))
        # print(self.MGRx.tck)

    def save(self, filnam):
        pklfile = open(filnam, "wb")
        pickle.dump(self, pklfile)
        pklfile.close()

    # load groundtrack from file
    def load(self, filnam):

        if os.path.isfile(filnam):
            pklfile = open(filnam, 'rb')
            self = pickle.load(pklfile)
            pklfile.close()
        else:
            print("No "+filnam+" found")
            self = None
        # print('Groundtrack loaded from '+filnam)
        # print(self.ladata_df)
        # print(self.MGRx.tck)

        return self

    def read_fill(self, infil):

        df = pd.read_csv(infil, sep=',', header=0)
        # print(df)
        # exit()
        df['orbID'] = infil.split('.')[0][-10:]
        self.name = df['orbID'].unique().squeeze()

        # strip and lower case all column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.lower()

        # only select the required data (column)
        self.df_input = df.copy()
        if (debug):
            df = df.loc[:, ['ephemeristime', 'tof_ns_et', 'chn', 'orbid', 'seqid', 'geoc_long', 'geoc_lat', 'altitude']]
        else:
            df = df.loc[:, ['ephemeristime', 'tof_ns_et', 'chn', 'orbid', 'seqid']]

        #pd.set_option('display.max_columns', 500)
        #pd.set_option('display.max_rows', 500)
        df.chn = pd.to_numeric(df.chn, errors='coerce').fillna(8).astype(np.int64)

        # remove bad data (chn > 4)
        df = df[df['chn'] < 5]
        # to compare to Mike's selection (chn >= 5 only ... )
        # df = df[df['EphemerisTime']>415023259.3]

        #df = df.drop(['chn'], axis=1)  # .set_index('seqid')

        # Reorder columns and reindex
        if debug:
            df.columns = ['ET_TX', 'TOF', 'chn', 'orbID', 'seqid', 'geoc_long', 'geoc_lat', 'altitude']
        else:
            df.columns = ['ET_TX', 'TOF', 'chn', 'orbID', 'seqid']

        #df = df.reset_index(drop=True)
        # print(df.index.is_unique)

        # Drop doublons from df, keeping the best chn for each observation
        doublons = df.sort_values(['ET_TX','chn']).loc[df.round(3).duplicated(['ET_TX'],keep='first')].index
        df.drop(doublons, inplace=True)
        df = df.reset_index(drop=True)

        # Convert TOF to seconds
        df.TOF *= 1.e-9

        # copy cleaned data to attribute df
        self.ladata_df = df

    def interpolate(self):

        # Read required trajectories from spice and interpolate
        startSpInterp = time.time()

        self.MGRx = interp_obj('MGRx')
        self.MGRv = interp_obj('MGRv')
        self.MGRa = interp_obj('MGRa')
        self.MERx = interp_obj('MERx')
        self.MERv = interp_obj('MERv')
        self.SUNx = interp_obj('SUNx')

        tstep = 1

        # Define call times for the SPICE
        try:
            t_spc = np.array(
                [x for x in np.arange(self.ladata_df['ET_TX'].min(), self.ladata_df['ET_TX'].max(), tstep)])
        except:
            print("*** ground_track.py: Issue interpolating ..." + self.name)
            print(self.ladata_df)
            print(self.ladata_df['ET_TX'].min())
            print(self.ladata_df['ET_TX'].max())
            exit(2)

        # print("Start spkezr MGR")

        # trajectory
        xv_spc = np.array([spice.spkezr(self.vecopts['SCNAME'],
                                        t,
                                        self.vecopts['INERTIALFRAME'],
                                        'NONE',
                                        self.vecopts['INERTIALCENTER']) for t in t_spc])[:, 0]

        xv_spc = np.reshape(np.concatenate(xv_spc), (-1, 6))

        # print("Start pxform MGR")
        # attitude
        pxform_array = np.frompyfunc(spice.pxform, 3, 1)
        cmat = pxform_array('MSGR_SPACECRAFT', self.vecopts['INERTIALFRAME'], t_spc)
        m2q_array = np.frompyfunc(spice.m2q, 1, 1)
        quat = m2q_array(cmat)
        quat = np.reshape(np.concatenate(quat), (-1, 4))

        # print("Start MGR interpolation")

        self.MGRx.interp([xv_spc[:, i] for i in range(0, 3)], t_spc)
        self.MGRv.interp([xv_spc[:, i] for i in range(3, 6)], t_spc)
        self.MGRa.interp([quat[:, i] for i in range(0, 4)], t_spc)

        # print("Start spkezr MER")

        xv_pla = np.array([spice.spkezr(self.vecopts['PLANETNAME'],
                                        t,
                                        self.vecopts['INERTIALFRAME'],
                                        'NONE',
                                        self.vecopts['INERTIALCENTER']) for t in t_spc])[:, 0]
        xv_pla = np.reshape(np.concatenate(xv_pla), (-1, 6))

        # print("Start MER interpolation")

        self.MERx.interp([xv_pla[:, i] for i in range(0, 3)], t_spc)
        self.MERv.interp([xv_pla[:, i] for i in range(3, 6)], t_spc)

        # print("Start spkezr SUN")

        xv_sun = np.array([spice.spkezr('SUN',
                                        t,
                                        self.vecopts['INERTIALFRAME'],
                                        'NONE',
                                        self.vecopts['INERTIALCENTER']) for t in t_spc])[:, 0]
        xv_sun = np.reshape(np.concatenate(xv_sun), (-1, 6))

        # print("Start SUN interpolation")

        self.SUNx.interp([xv_sun[:, i] for i in range(0, 3)], t_spc)

        # save to orbit-wise file
        self.SpObj = {'MGRx': self.MGRx,
                      'MGRv': self.MGRv,
                      'MGRa': self.MGRa,
                      'MERx': self.MERx,
                      'MERv': self.MERv,
                      'SUNx': self.SUNx}

        pickleIO.save(self.SpObj, auxdir + 'spaux_' + self.name + '.pkl')

        endSpInterp = time.time()
        if (debug):
            print('----- Runtime SpInterp = ' + str(endSpInterp - startSpInterp) + ' sec -----' + str(
                (endSpInterp - startSpInterp) / 60.) + ' min -----')

    def geoloc(self, get_partials=partials):

        #vecopts = self.vecopts.copy()

        #################### end -------

        # ------------------------------
        # Processing
        # ------------------------------

        # Compute geolocalisation and dxyz/dP, where P = (A,C,R,Rl,Pt)
        startGeoloc = time.time()

        # Prepare
        if get_partials:
            param = {'': 1.}
            param.update(parOrb)
            param.update(parGlo)
        # \
        # 'dh2':0.1}
        else:
            param = {'': 1.}

        self.param = param
        # check if track has to be perturbed (else only apply global pars)
        if self.name in pert_tracks or pert_tracks == []:
            _ = {}
            # for k, v in pert_cloop.items():
            [_.update(v) for k, v in pert_cloop.items()]
            self.pert_cloop = _.copy()
        else:
            self.pert_cloop = pert_cloop['glo'].copy()
        
        if debug:
            print('check pert', self.name, self.pert_cloop)

        # read solution from previous iteration and
        # add to self.pert_cloop (orb and glo)
        if self.sol_prev_iter != None:
            self.par_solupd()

        if debug:
            print('check pert', self.name, self.pert_cloop)

        if hasattr(self, 'SpObj'):
            SpObj = self.SpObj
        else:
            SpObj = {'MGRx': self.MGRx,
                     'MGRv': self.MGRv,
                     'MGRa': self.MGRa,
                     'MERx': self.MERx,
                     'MERv': self.MERv,
                     'SUNx': self.SUNx}
        #########################

        if (
                parallel and SpInterp > 0 and 1 == 2):  # spice is not multi-thread (yet). Could be improved by fitting a polynomial to
            # the orbit (single initial call) with an appropriate accuracy.
            # print((mp.cpu_count() - 1))
            pool = mp.Pool(processes=mp.cpu_count() - 1)
            results = pool.map(self.get_geoloc_part, param.items())  # parallel
            pool.close()
            pool.join()
        else:
            results = [self.get_geoloc_part(i) for i in param.items()]  # seq

        # exit()

        # store ladata_df for update
        ladata_df = self.ladata_df.copy()

        if (self.vecopts['OUTPUTTYPE'] == 0):
            ladata_df['X'] = results[0][:, 0]
            ladata_df['Y'] = results[0][:, 1]
            ladata_df['Z'] = results[0][:, 2]
            # if sim:
            #   _, _, Rbase = subprocess.check_call([PGM_HOME+'diff_res_format', d+'/resid.asc', d_part+'/resid.asc', dif_dir+'/diff.resid_'+d],
            #           universal_newlines=True)
            # else:
            Rbase = self.vecopts['PLANETRADIUS'] * 1.e3
            ladata_df['R'] = np.linalg.norm(results[0], axis=1) - Rbase

        elif (self.vecopts['OUTPUTTYPE'] == 1):
            ladata_df['LON'] = results[0][:, 0]
            ladata_df['LAT'] = results[0][:, 1]
            #print(len(results[0]))
            Rbase = self.vecopts['PLANETRADIUS'] * 1.e3
            ladata_df['R'] = results[0][:, 2] - Rbase

        if debug:
          print(ladata_df)
          print(results[0][0,:],list(param)[0],len(param))

        if (len(param) > 1 and list(param)[0] == ''):
            if (self.vecopts['OUTPUTTYPE'] == 0):
                for i in range(1, len(param)):
                    ladata_df['dX/' + list(param)[i]] = results[i][:, 0]
                    ladata_df['dY/' + list(param)[i]] = results[i][:, 1]
                    ladata_df['dZ/' + list(param)[i]] = results[i][:, 2]
                    ladata_df['dR/' + list(param)[i]] = np.linalg.norm(results[i], axis=1)

                # Add partials w.r.t. tidal h2
                ladata_df['dLON/dh2'] = 0
                ladata_df['dLAT/dh2'] = 0
                ladata_df['dR/dh2'] = \
                tidepart_h2(self.vecopts, np.hstack([ladata_df['X'], ladata_df['Y'], ladata_df['Z']]), \
                            ladata_df['ET_TX'] + 0.5 * ladata_df['TOF'], SpObj)[0]

            elif (self.vecopts['OUTPUTTYPE'] == 1):
                for i in range(1, len(param)):
                    ladata_df['dLON/' + list(param)[i]] = results[i][:, 0]
                    ladata_df['dLAT/' + list(param)[i]] = results[i][:, 1]
                    ladata_df['dR/' + list(param)[i]] = results[i][:, 2]

                # Add partials w.r.t. tidal h2
                parGlo['dh2'] = 1
                param['dh2'] = 1
                self.vecopts['PARTDER'] = ''
                ladata_df['dLON/dh2'] = 0
                ladata_df['dLAT/dh2'] = 0
                ladata_df['dR/dh2'] = tidepart_h2(self.vecopts, \
                                                  np.transpose(astr.sph2cart(
                                                      ladata_df['R'].values + self.vecopts['PLANETRADIUS'] * 1.e3,
                                                      ladata_df['LAT'].values, ladata_df['LON'].values)), \
                                                  ladata_df['ET_TX'].values + 0.5 * ladata_df['TOF'].values, SpObj)[0]

        # update object attribute df
        self.ladata_df = ladata_df.copy()

        if debug:
            print(ladata_df)
            exit()

        endGeoloc = time.time()
        if (debug):
            print('----- Runtime Geoloc = ' + str(endGeoloc - startGeoloc) + ' sec -----' + str(
                (endGeoloc - startGeoloc) / 60.) + ' min -----')

    def par_solupd(self):
        """
        Updates perturbations to a priori parameters based on solution from previous iteration (stored at AccumXov step
        in pkl file).
        """
        if len(self.sol_prev_iter['orb']) > 0:
            corr_orb = self.sol_prev_iter['orb'].filter(regex='sol.*$', axis=1).apply(pd.to_numeric, errors='ignore',
                                                                                      downcast='float'
                                                                                      ).to_dict('records')[0]
            corr_orb = {key.split('_')[1].split('/')[1]: corr_orb[key] for key in corr_orb.keys()}
            self.pert_cloop = mergsum(self.pert_cloop, corr_orb)

        if len(self.sol_prev_iter['glo']) > 0:
            _ = self.sol_prev_iter['glo'].apply(pd.to_numeric, errors='ignore', downcast='float')
            corr_glo = dict(zip(_.par, _.sol))
            corr_glo = {key.split('/')[1]: corr_glo[key] for key in corr_glo.keys()}

            self.pert_cloop = mergsum(self.pert_cloop, corr_glo)

        if debug:
            print(self.pert_cloop)

    # @profile
    def get_geoloc_part(self, par):

        tmp_df = self.ladata_df.copy()
        #vecopts = self.vecopts
        if hasattr(self, 'SpObj'):
            SpObj = self.SpObj
        else:
            SpObj = {'MGRx': self.MGRx,
                     'MGRv': self.MGRv,
                     'MGRa': self.MGRa,
                     'MERx': self.MERx,
                     'MERv': self.MERv,
                     'SUNx': self.SUNx}

        # get dictionary values
        self.vecopts['PARTDER'] = list(par)[0]
        diff_step = list(par)[1]

        if (debug):
            print('geoloc: ' + str(par))
            print('self.vecopts[PARTDER]', self.vecopts['PARTDER'])
            print(diff_step)

        # Read self.vecopts[partder] and apply perturbation
        # if needed (for partials AND for closed loop sim)
        tmp_pertPar = self.pertpar(diff_step)

        # tmp_pertPar = {**tmp_pertPar, **self.pert_cloop}
        # print('norm',tmp_pertPar, diff_step)
        # Get bouncing point location (XYZ or LATLON depending on self.vecopts)
        geoloc_out, et_bc = geoloc(tmp_df, self.vecopts, tmp_pertPar, SpObj, t0 = self.t0_orb)

        #print(self.ladata_df)
        tmp_df.loc[:,'ET_BC'] = et_bc

        # Compute partial derivatives if required
        if self.vecopts['PARTDER'] is not '':

            # Read self.vecopts[partder] and apply perturbation
            # if needed (for partials AND for closed loop sim)
            tmp_pertPar = self.pertpar(diff_step, -1.)

            # print('part',tmp_pertPar, diff_step)
            geoloc_min, et_bc = geoloc(tmp_df, self.vecopts, tmp_pertPar, SpObj)
            partder = (geoloc_out[:, 0:3] - geoloc_min[:, 0:3])

            tmp_df.loc[:,'ET_BC_'+self.vecopts['PARTDER']] = et_bc
            #print(self.ladata_df)
            #exit()

            # print("partder check", geoloc_out[:,0:3], geoloc_min[:,0:3],diff_step)

            if (self.vecopts['PARTDER'] in ('dA', 'dC', 'dR', 'dRl', 'dPt')):
                partder /= (2. * diff_step)  # [0]
                # print('partder', partder)
            else:
                # print('norm_pert', np.linalg.norm(diff_step))
                partder /= 2. * np.linalg.norm(diff_step)  # [1]
                # print('partder', partder)

            self.ladata_df = tmp_df

            return partder

        else:
            self.ladata_df = tmp_df

            return geoloc_out

    def pertpar(self, diff_step, sign=1.):
        tmp_pertPar = self.pertPar.copy()
        tmp_pertcloop = self.pert_cloop.copy()
        # set all elements to 0 before reworking
        tmp_pertPar = tmp_pertPar.fromkeys(tmp_pertPar, 0)

        tmp_pertPar[self.vecopts['PARTDER']] = diff_step

        if sign != 1:
            try:
                tmp_pertPar[self.vecopts['PARTDER']] *= sign
            except:  # if perturbation is a vector
                tmp_pertPar[self.vecopts['PARTDER']] = [sign * x for x in tmp_pertPar[self.vecopts['PARTDER']]]

        # add pert for closed loop sim
        for key in tmp_pertcloop:
            if key in tmp_pertPar:

                if debug:
                    print("pertpar", self.name, key, tmp_pertPar[key], tmp_pertcloop[key])

                np.random.seed(int(self.name))
                tmp_pertcloop[key] = tmp_pertcloop[key]*np.random.randn()

                if hasattr(tmp_pertPar[key], "__len__"):
                    tmp_pertPar[key] = [sum(pair) for pair in zip(tmp_pertPar[key], tmp_pertcloop[key])]
                else:
                    tmp_pertPar[key] += tmp_pertcloop[key]
                # print(key, len(tmp_pertPar[key]),tmp_pertPar[key])

        self.pertPar = tmp_pertPar.copy()
        return tmp_pertPar

    # Compute stereographic projection of measurements location
    # and feed it to ladata_df
    def project(self,lon0=0,lat0=90,inplace=True):

        ladata_df = self.ladata_df.copy()
        param = self.param

        if partials==0:
            param = dict([next(iter(param.items()))])

        startProj = time.time()

        if (parallel and False):  # not convenient for small datasets (<5 orbits)
            # print((mp.cpu_count() - 1))
            pool = mp.Pool(processes=mp.cpu_count() - 1)
            results = pool.map(self.launch_stereoproj, param.items())  # parallel
            # ladata_df = ladata_df.join(pd.DataFrame(results, index=ladata_df.index))
            # print(ladata_df)
            # exit()
            pool.close()
            pool.join()
        else:
            results = [self.launch_stereoproj(i,lon0,lat0) for i in param.items()]  # seq

        # print(results)

        # prepare column names
        col_sim = np.array(['X_stgprj', 'Y_stgprj'])
        if partials:
            col_der = np.hstack([['X_stgprj_' + par + '_m', 'Y_stgprj_' + par + '_m', 'X_stgprj_' + par + '_p',
                                  'Y_stgprj_' + par + '_p'] for par in list(param.keys())[1:]])
            # concatenate stgprj to ladata_df
            ladata_df[np.hstack((col_sim, col_der))] = pd.DataFrame(np.hstack(results), index=ladata_df.index)
        else:
            ladata_df[np.hstack((col_sim))] = pd.DataFrame(np.hstack(results), index=ladata_df.index)

        if inplace:
            self.ladata_df = ladata_df.copy()

        endProj = time.time()
        if (debug):
            print('----- Runtime Proj = ' + str(endProj - startProj) + ' sec -----' + str(
                (endProj - startProj) / 60.) + ' min -----')

        return ladata_df

    # @profile
    #################
    # Correct for perturbations by using partials (if needed) and launch projection
    def launch_stereoproj(self, par_d, lon0 = 0, lat0 = 90):

        ladata_df = self.ladata_df
        vecopts = self.vecopts

        #print('proj: ' + str(par_d))

        # get dict values
        par = list(par_d)[0]
        diff_step = np.linalg.norm(list(par_d)[1])

        # setting up as 2D arrays to be able to access np.shape(lon_tmp)[0]
        # in the case par='' (no partial derivative)
        lon_tmp = [ladata_df['LON'].values]
        lat_tmp = [ladata_df['LAT'].values]

        # Apply position correction
        if (par is not ''):
            lon_tmp = [lon_tmp[:][0] + ladata_df['dLON/' + par].values * k * diff_step for k in [-1, 1]]
            lat_tmp = [lat_tmp[:][0] + ladata_df['dLAT/' + par].values * k * diff_step for k in [-1, 1]]

            if (debug):
                print('corr: lon lat', lon_tmp[:][0], lat_tmp[:][0])
                print('corr: partials add', ladata_df['dLAT/' + par].values * diff_step, ladata_df['dLON/' + par].values * diff_step)

        # project latlon to xy from North Pole in stereo projection
        proj = np.vstack([project_stereographic(lon_tmp[:][k], lat_tmp[:][k], lon0, lat0, vecopts['PLANETRADIUS']) for k in
                          range(0, np.shape(lon_tmp)[0])]).T

        # stg_proj_cols = {}

        if (par is ''):
            # proj=project_stereographic(lon_tmp,lat_tmp,0,90,2440)
            # ladata_df['X_stgprj']=proj[:,0]
            # ladata_df['Y_stgprj']=proj[:,1]
            # stg_proj_cols=proj[:,0:2]

            # print(np.array(proj).reshape(-1,2))
            return np.array(proj).reshape(-1, 2)  # stg_proj_cols


        else:
            # setup column names
            # stg_proj_cols.update({'X_stgprj_'+par+'_p':None,'Y_stgprj_'+par+'_p':None,'X_stgprj_'+par+'_m':None,'Y_stgprj_'+par+'_m':None})
            # store proj XY with pos/neg pert
            # for idx in range(4):
            # stg_proj_cols=proj[:,0:4]
            # ladata_df[name]=proj[:,idx]

            # print(np.array(proj).reshape(-1,4))

            return np.array(proj).reshape(-1, 4)  # stg_proj_cols

#################
