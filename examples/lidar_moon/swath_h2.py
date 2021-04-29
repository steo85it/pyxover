import glob
import os
import sys

import numpy as np
import pandas as pd
import spiceypy as sp
from matplotlib import pyplot as plt
# from astropy import units as u
#
# from poliastro.bodies import Moon
# from poliastro.twobody import Orbit
from tqdm import tqdm

# from examples.MLA.options import OrbRep

from config import XovOpt
from src.accumxov import AccumXov

basedir = ""
#basedir = "/home/sberton2/Works/NASA/Mercury_tides/pyxover_release/examples/lidar_moon/"
generate_data = False
remove_poles = False # True

if len(sys.argv) > 2:
    orb_freq = sys.argv[1]
    dlon_in = f'{int(sys.argv[2]):02d}'
    print(f"Running with: orb_freq={orb_freq} and dlon_in={dlon_in}...")
else:
    print(f"Use as python swath_h2.py dlon_in orb_freq_H.")
    orb_freq = '24H'
    dlon_in = '30'
    print(f"Running with standard arguments orb_freq={orb_freq} and dlon_in={dlon_in}...")

def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r

def sph2cart(azimuth,elevation,r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def gen_simdata():
    # global ax, df_geotrg, df_xovers
    # %% input parameters
    dlat = 15
    dlon = 5
    offmax = 5
    # %% build grid of geodetic targets (to monitor tide)
    lat_ = list(range(-90, 90, dlat))
    lat_ = np.sort(lat_[1:] + [-87.5, 87.5])
    lon_ = np.arange(dlon / 2, (360 - 1e-5), dlon)
    lon__, lat__ = np.meshgrid(lon_, lat_)
    if False:
        plt.plot(lon__, lat__, marker='.', color='k', linestyle='none')
        plt.show()
    x__, y__, z__ = sph2cart(lon__ / 180 * np.pi, lat__ / 180 * np.pi, 1737.4)
    if False:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x__, y__, z__, rstride=1, cstride=1,
                        cmap='terrain', edgecolor=None)
        ax.set(xlabel="x", ylabel="y", zlabel="f(x, y)", title='title')
        plt.show()
    # assign epochs in range to geodetic points (randomly)
    sp.furnsh('/home/sberton2/Works/NASA/radiosity/py-flux-app/examples/simple.furnsh')
    utc0 = '2010-09-15-00:00:00.00'
    et0 = sp.str2et(utc0)
    utc1 = '2011-09-15-00:00:00.00'
    et1 = sp.str2et(utc1)
    print(et0, et1)
    df_geotrg = pd.DataFrame(np.transpose([lon__.ravel(), lat__.ravel()]), columns=['LON', 'LAT'])
    df_geotrg['x'] = x__.ravel()
    df_geotrg['y'] = y__.ravel()
    df_geotrg['z'] = z__.ravel()
    df_geotrg.reset_index(inplace=True)
    df_geotrg = df_geotrg.rename({'index': 'geotrg'}, axis='columns')
    print(df_geotrg)
    # generate random series of epochs and geotrg
    num_tracks = 3
    num_xovers = 8
    rand_ = np.random.random_sample((num_xovers,))
    track1 = np.floor(rand_ * num_tracks)
    t1 = rand_ * (et1 - et0) + et0
    rand_ = np.random.random_sample((num_xovers,))
    track2 = np.floor(rand_ * num_tracks)
    t2 = rand_ * (et1 - et0) + et0
    geotrg = np.floor(rand_ * len(df_geotrg))
    df_xovers = pd.DataFrame(np.transpose([track1, track2, t1, t2, geotrg]),
                             columns=['track1', 'track2', 't1', 't2', 'geotrg'])
    df_xovers['std'] = 1.  # rand_
    df_xovers['dR'] = 10
    df_xovers['R_A'] = 10 + rand_
    df_xovers['R_B'] = rand_

    return df_geotrg, df_xovers

#@profile
def convert_xovers_input(df_geotrg, df_xovers, outname):

    df_geotrg['xov_count'] = df_xovers.groupby('geotrg').count().iloc[:, 0]

    df_xovers = df_xovers.merge(df_geotrg[['LON', 'LAT', 'geotrg']], on='geotrg')
    # print(df_xovers.loc[df_xovers.geotrg==456])
    df_xovers['std'] = 1. / (df_xovers['std'].values**2)

    # print(df_xovers.columns)
    df_xovers = df_xovers.rename({'std': 'weights'}, axis='columns')
    df_xovers = df_xovers.astype({'orbA': 'int32', 'orbB': 'int32'})
    # df_xovers.columns = ['orbA','orbB','dtA','dtB','xOvID','weights','dR','R_A','R_B','LON','LAT']
    xovers_cols = ['LAT', 'LON', 'R_A', 'R_B', 'cmb_idA', 'cmb_idB', 'dR',
                   'dR/dR_A', 'dR/dR_B', 'dR/dh2', 'dist_Am', 'dist_Ap', 'dist_Bm',
                   'dist_Bp', 'dist_max', 'dist_min_mean', 'dtA', 'dtB', 'huber',
                   'huber_trks', 'interp_weight', 'mla_idA', 'mla_idB', 'offnad_A',
                   'offnad_B', 'orbA', 'orbB', 'weights', 'x0', 'y0', 'xOvID']
    df_xovers_transf = pd.DataFrame(np.zeros((len(df_xovers), len(xovers_cols))), columns=xovers_cols)
    df_xovers_transf += df_xovers
    # df_xovers.dR = 0.
    df_xovers_transf.fillna(1, inplace=True)
    df_xovers_transf = df_xovers_transf.astype({'orbA': 'int32', 'orbB': 'int32', 'xOvID': 'int32'})
    df_xovers_transf['dR/dh2'] = df_xovers.dzA - df_xovers.dzB
    print("df mean,std,med: ", df_xovers_transf.weights.mean(),
          df_xovers_transf.weights.std(), df_xovers_transf.weights.median())

    # remove poles
    if remove_poles:
        print(f'pre len_xovers:{len(df_xovers_transf)}')
        df_xovers_transf = df_xovers_transf.loc[df_xovers_transf.LAT.abs() >= 87.5]
        print(f'post len_xovers:{len(df_xovers_transf)}')

    from src.pyxover.xov_setup import xov
    # from examples.MLA.options import vecopts
    simxov = xov(XovOpt.get("vecopts"))
    simxov.xovers = df_xovers_transf

    simxov.parOrb_xy = ['dR/dR_A', 'dR/dR_B']
    simxov.parGlo_xy = ['dR/dh2']
    # Retrieve all orbits involved in xovers
    orb_unique = simxov.xovers['orbA'].tolist()
    orb_unique.extend(simxov.xovers['orbB'].tolist())
    simxov.tracks = [str(x) for x in set(orb_unique)]
    simxov.save(
        f'{outname}')
    print(f"Generated xovers table with {len(simxov.xovers)} rows and {len(simxov.tracks)} tracks/parameters (+1 for h2)!")

    return df_geotrg, simxov.xovers

#@profile
def convert_simdata(input_xovers_file):

    df_xovers = pd.read_csv(input_xovers_file, sep='\s+', comment='#')
    df_xovers.columns = ['xOvID', 'geotrg', 'dtA', 'dtB', 'dzA', 'dzB']
    df_xovers[['xOvID', 'geotrg']] -= 1

    # generate random value for xovers discrepancies
    rand_ = np.random.random_sample((len(df_xovers),))
    df_xovers['std'] = 5.e-5  #  # rand_
    df_xovers['dR'] = 0.
    df_xovers['R_A'] = 0. #+ rand_
    df_xovers['R_B'] = 0. #+ rand_

    j2000_unixt = 946684800.

    df_xovers = df_xovers.iloc[:]
    for orb in ['A', 'B']:
        df_xovers[f't{orb}_ts'] = pd.to_datetime(j2000_unixt + df_xovers[f'dt{orb}'], unit='s')
        df_xovers = df_xovers.sort_values(by=f't{orb}_ts').reset_index(drop=True)

        gb = df_xovers.groupby([pd.Grouper(key=f't{orb}_ts', freq=orb_freq)])
        df_xovers[f'orb{orb}'] = pd.Series({v: k.strftime('%y%m%d%H%M') for k, V in gb.indices.items() for v in V})
        df_xovers[f't{orb}_0'] = pd.Series({v: k for k, V in gb.indices.items() for v in V})

        df_xovers[f'dt{orb}'] = (df_xovers[f't{orb}_ts'] - df_xovers[f't{orb}_0']).dt.total_seconds()
        df_xovers.drop([f't{orb}_ts', f't{orb}_0'], inplace=True, axis=1)

    df_xovers = df_xovers.sort_values(by='xOvID').reset_index(drop=True)

    return df_xovers

def import_landmarks(input_geotrg_file):
    df_geotrg = pd.read_csv(input_geotrg_file, sep='\s+')
    df_geotrg = df_geotrg.rename({'#LMKID': 'geotrg'}, axis='columns')
    df_geotrg.geotrg -= 1

    return df_geotrg


def plot_std(df_, fname, df_geotrg, unconstr = False):

    npars = dict(zip(range(len(pd.unique(df_['par']))),pd.unique(df_['par'])))

    fig, axs = plt.subplots(len(npars.keys())+1)

    # if True:
    axs[0].set_title(list(zip(['dlon','dlat','offmax'],get_numbers_from_str(fname)[1:])), fontsize=15)
    df_geotrg.plot.scatter(x="LON", y="LAT", c="xov_count", s=0.5, cmap='viridis',ax=axs[0])  # , s=50)
    # plt.show()

    # df = pd.DataFrame(par_names.items())
    # df['std'] = std_pars
    # df.columns = ['par', 'npar', 'std']
    # # TODO very custom
    # df['par'] = [int(x.split('_')[0]) for x in df['par'].values[:-1]] + [0]
    # df_ = df.iloc[:-1].sort_values(by='par').reset_index(drop=True).reset_index()
    # print(df_)

    # axs[1].set_title(r'$\sigma_{h_2}=$' + f'{100. * std_pars[-1]:.2f}%', fontsize=15)
    for idx,par in npars.items():
        df_cut = df_.loc[df_['par']==par]
        df_cut.plot.scatter(x='npar', y='std', c='nobs', cmap='viridis', ax=axs[idx+1])
        # plt.scatter(list(par_names.keys())[:-1],std_pars[:-1])
        # axs[idx+1].set_xlabel('track #')
        if par in ['dA','dC','dR']:
            unit_par = 'km'
        if par in ['dA1', 'dC1', 'dR1']:
            unit_par = 'km/d'
        if par in ['dA2','dC2','dR2']:
            unit_par = 'km/d^2'
        if par in ['dRC','dRS']:
            unit_par = 'km'

        axs[idx+1].set_ylabel(r'$\sigma$'+f'({par}), {unit_par}')

    figname = fname.split('/')[-1].split('.')[0]

    plt.tight_layout()
    if unconstr == False:
        plt.savefig(f"{basedir}{figname}.png")
    else:
        plt.savefig(f"{basedir}{figname}_unconstr.png")

def get_numbers_from_str(fname):
    name_to_parse = fname.split('/')[-1].split('.')[0]
    newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in name_to_parse)
    dlon_dlat_offmax_ = [float(i) for i in newstr.split()]
    return dlon_dlat_offmax_


def count_obs_per_orb(df_xovers):
    nobs_tracks = df_xovers[['orbA', 'orbB']].apply(pd.Series.value_counts).sum(
        axis=1).sort_values(
        ascending=False)
    return nobs_tracks


def prepare_accum_out(par_names, std_pars, df_obs_per_orb):
    df = pd.DataFrame(par_names.items())
    df['std'] = std_pars
    df.columns = ['orbpar', 'npar', 'std']
    df['orb'] = [int(x.split('_')[0]) for x in df['orbpar'].values[:-1]] + [0]
    df['par'] = [x.split('_')[1].split('/')[-1] for x in df['orbpar'].values[:-1]] + [0]

    df_ = df.iloc[:-1].sort_values(by='par').reset_index(drop=True).reset_index()

    df_obs_per_orb.reset_index(inplace=True)
    df_obs_per_orb['orb'] = [int(x) for x in df_obs_per_orb['index'].values[:]]

    df_ = df_.merge(df_obs_per_orb, left_on='orb', right_on='orb', how='inner')
    df_ = df_[['orb','par', 'std', 'nobs']].reset_index()
    df_ = df_.rename({'index':'npar'},axis=1)
    return df_

#@profile
def main():


    if generate_data:
        df_geotrg, df_xovers = gen_simdata()
    else:
        input_dir = f"{basedir}data/raw/"
        print(f'{input_dir}crossovers_21012?_dlon{dlon_in}_*.txt',f'{input_dir}landmarks_21012?_dlon{dlon_in}.txt')
        input_data = glob.glob(f'{input_dir}crossovers_21012?_dlon{dlon_in}_*.txt')
        input_landmarks = glob.glob(f'{input_dir}landmarks_21012?_dlon{dlon_in}.txt')[0]

        #input_data = glob.glob(f'/home/sberton2/tmp/testLIDAR/crossovers_21012?_dlon{dlon_in}_*.txt')
        print(input_data)
        #input_landmarks = glob.glob(f'/home/sberton2/tmp/testLIDAR/landmarks_21012?_dlon{dlon_in}.txt')[0]
        print(input_landmarks)

        df_geotrg = import_landmarks(input_geotrg_file=input_landmarks)
        error_orbs_rms = []
        error_h2 = []
        dlon_dlat_offmax = []
        for idx, fname in tqdm(enumerate(input_data[:])):

            # prepare directories
            soldir = f"BS{idx}_0"
            xovdir = f'{basedir}data/out/sim/{soldir}/0res_1amp/xov/'
            os.makedirs(xovdir, exist_ok=True)
            if remove_poles:
                outdir = f'{basedir}out/{orb_freq}/{XovOpt.get("OrbRep")}_nopole/'
            else:
                outdir = f'{basedir}out/{orb_freq}/{XovOpt.get("OrbRep")}/'

            os.makedirs(outdir, exist_ok=True)

            # parse filename for setup
            dlon_dlat_offmax_ = get_numbers_from_str(fname)[1:]

            df_xovers = convert_simdata(input_xovers_file=input_data[idx])

            df_obs_per_orb = count_obs_per_orb(df_xovers).to_frame('nobs')

            df_geotrg_, df_xovers = convert_xovers_input(df_geotrg=df_geotrg.copy(), df_xovers=df_xovers,
                                                         outname=xovdir + 'xov_00_00.pkl')

            # for idx in tqdm(range(len(input_data))):
            outdirnams = [f'sim/{soldir}/0res_1amp/']
            data_sim = 'sim'
            ext_iter = 0
            # try:
            par_names, std_pars, std_pars_unconstrained = AccumXov.main([outdirnams, data_sim, ext_iter])
            # plot constrained solution
            df_ = prepare_accum_out(par_names, std_pars,df_obs_per_orb)
            plot_std(df_, fname, df_geotrg_)
            # plot unconstrained solution
            df_ = prepare_accum_out(par_names, std_pars_unconstrained,df_obs_per_orb)
            plot_std(df_, fname, df_geotrg_, unconstr=True)

            dlon_dlat_offmax.append(dlon_dlat_offmax_)
            error_orbs_rms.append(np.sqrt(np.mean(std_pars[:-1] ** 2)))
            error_h2.append((std_pars[-1]))
            # except:
            #     print(f"Failed on iter {idx} and file {fname}... Next!")

        print(dlon_dlat_offmax)
        print(np.vstack([error_orbs_rms,error_h2]).T)
        df_results = pd.DataFrame(np.hstack([dlon_dlat_offmax,np.vstack([error_orbs_rms,error_h2]).T]))
        df_results.columns = ['dlon','dlat','doffmax','rms_orb_err','h2_err']
        df_results['rms_orb_err'] *= 1.e3
        print(df_results)
        df_results.to_pickle(f'{outdir}df_results_dlon{dlon_in}.pkl')
        df_results.to_csv(f'{outdir}df_results_dlon{dlon_in}.csv')

        return df_results, df_xovers

if __name__ == '__main__':

    # update paths and check options
    XovOpt.set("body", 'MOON')  #
    XovOpt.set("instrument", "pawstel")  # needs to be set directly in config, too, else error (accum_opt doesn't see the updated value)
    XovOpt.set("basedir", 'data/')
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
    XovOpt.set("OrbRep", 'per')

    XovOpt.set("sol4_orb", [])
    XovOpt.set("sol4_orbpar", ['dRC','dRS']) # None] #['dA','dC', #,'dRl','dPt'] #] #
    XovOpt.set("sol4_glo", ['dR/dh2'])

    XovOpt.set("par_constr", {'dR/dRA': 1.e2, 'dR/dDEC': 1.e2, 'dR/dL': 1.e2, 'dR/dPM': 1.e2, 'dR/dh2': 5.e0, 'dR/dA': 1.e2,
                  'dR/dC': 1.e2, 'dR/dR': 5.e-3,  # } #, 'dR/dRl':5.e1, 'dR/dPt':5.e1} #
                  'dR/dA1': 1.e-1, 'dR/dC1': 1.e-1, 'dR/dR1': 1.e2, 'dR/dA2': 1.e-2, 'dR/dC2': 1.e-2, 'dR/dR2': 1.e2,
                  'dR/dAC': 1.e-1, 'dR/dCC': 1.e-1, 'dR/dRC': 1.e2, 'dR/dAS': 1.e-2, 'dR/dCS': 1.e-2,
                  'dR/dRS': 1.e2})  # , 'dR/dA2':1.e-4, 'dR/dC2':1.e-4,'dR/dR2':1.e-2} # 'dR/dA':100., 'dR/dC':100.,'dR/dR':100.} #, 'dR/dh2': 1} #
    XovOpt.set("mean_constr", {'dR/dA': 1.e0, 'dR/dC': 1.e0, 'dR/dR': 5.e-2})

    XovOpt.check_consistency()

    main()
