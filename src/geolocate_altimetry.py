#!/usr/bin/env python3
# ----------------------------------
# geolocate_altimetry.py
#
# Description: Find latitude and longitude of
# altimetry beam crossing
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
# vecopts contains:
# SCID (e.g. -236)
# SCNAME (e.g. MESSENGER)
# SCFRAME (e.g. -236000)
# PLANETID (e.g. 199)
# PLANETNAME ( e.g. MERCURY)
# PLANETRADIUS (e.g. 2440.0)
# PLANETFRAME (e.g. IAU_MERCURY)
# OUTPUTTYPE = 0 for XYZ, 1 for LON/LAT/R
# ALTIM_BORESIGHT in SBF
# INERTIALFRAME 'J2000'
# INERTIALCENTER 'SSB'
import re

import numpy as np
import spiceypy as spice
from astropy import constants as const
from astropy.constants import c as clight

# mylib
from xovutil import astro_trans as astr
from xovutil.icrf2pbf import icrf2pbf
# from examples.MLA.options import XovOpt.get("SpInterp")
from config import XovOpt
from xovutil.orient_setup import orient_setup
from tidal_deform import tidal_deform


# from collections import defaultdict

##############################################
# #@profile
from xovutil.units import rad2as, as2rad, sec2day


def geoloc(inp_df, vecopts, tmp_pertPar, SpObj, t0 = 0):
    """

    :type inp_df: ladata_df containing TOF(sec) and ET_TX(sec from J2000)
    """
    #  ABCORR = 'NONE'
    # TODO check tof unit (sec or ns???)
    tof = inp_df['TOF'].values
    et_tx = inp_df['ET_TX'].values

    #print('tof')
    #print(tof)
    # print('et_tx')
    # print(et_tx)

    # print("geoloc",tmp_pertPar)
    # exit()

    oneway = tof * clight.value / 2.
    twoway = tof * clight.value

    # Set all corrections to 0
    # dACR = [0, 0, 0]

    #print("testObjGeoLoc", SpObj['MGRx'].eval(et_tx))
    #exit()

    scpos_tx, scvel_tx = get_sc_ssb(et_tx, SpObj, tmp_pertPar, vecopts, t0 = t0)
    # update after offset
    Rtx = np.linalg.norm(scpos_tx, axis=1)

    # get probe CoM state at RX
    # --------------------------
    scpos_rx, scvel_rx = get_sc_ssb(et_tx + tof, SpObj, tmp_pertPar, vecopts, t0 = t0)
    # update after offset
    Rrx = np.linalg.norm(scpos_rx, axis=1)

    # get planet barycenter state (SSB J2000) at bounce
    # --------------------------------------------------
    et_bc = et_tx + tof / 2.

    if (XovOpt.get("SpInterp") > 0):
        plapos_bc = np.transpose(SpObj['MERx'].eval(et_bc))
    else:
        plapos_bc, lt = spice.spkpos(vecopts['PLANETNAME'],
                                     et_bc,
                                     vecopts['INERTIALFRAME'],
                                     'NONE',
                                     vecopts['INERTIALCENTER'])

    plapos_bc = 1.e3 * np.array(plapos_bc)
    # print(plapos_bc)

    # compute SSB to bounce point vector
    if XovOpt.get("instrument") == 'BELA':
        # project tof along radial dir between s/c and planet (=nadir pointing)
        zpt = (-scpos_tx+plapos_bc)/(np.linalg.norm(scpos_tx-plapos_bc,axis=1)[:, np.newaxis])
    else:
        # get altimeter boresight in S/C frame
        zpt = np.tile(XovOpt.get("vecopts")['ALTIM_BORESIGHT'], np.size(scpos_tx, 0)).reshape(-1, 3)

        # compute s/c frame to inertial rotation (using np.frompyfunc to vectorize pxform)
        # ck+fk+sclk needed
        if (XovOpt.get("SpInterp") > 0):
            cmat = SpObj['MGRa'].evalCmat(et_tx)
        else:
            pxform_array = np.frompyfunc(spice.pxform, 3, 1)
            cmat = pxform_array(vecopts['SCFRAME'], vecopts['INERTIALFRAME'], et_tx)

        # rotate boresight dir to inertial frame
        zpt = [np.dot(cmat[i], zpt[i]) for i in range(0, np.size(zpt, 0))]
        # print(np.array(vmbf).reshape(-1,3))

    if ([tmp_pertPar[k] for k in ['dRl', 'dPt']] != [0, 0]):
        # Apply roll and pitch offsets to zpt (converted to radians)
        # print(np.reshape(np.tile([tmp_pertPar[k] for k in ['dRl','dPt']],len(et_tx)),(-1,2)))
        ang_Rl = np.reshape(np.tile([as2rad(tmp_pertPar[k]) for k in ['dRl', 'dPt']], len(et_tx)), (-1, 2))[:, 0]
        ang_Pt = np.reshape(np.tile([as2rad(tmp_pertPar[k]) for k in ['dRl', 'dPt']], len(et_tx)), (-1, 2))[:, 1]

        zpt = astr.rp_2_xyz(zpt, ang_Rl, ang_Pt)
    #    print("test zpt post dRlPt", np.linalg.norm(zpt, axis=1))

    # print(zpt,np.size(scpos_tx,0))

    # compute corrections to oneway tof - average of Shapiro delay on
    # each branch (!!! Ri, Rj, etc are w.r.t. SSB and not w.r.t. perturbing
    # body, which is wrong but probably acceptable)

    oneway = range_corr_iter(Rrx, Rtx, oneway, scpos_rx, scpos_tx, twoway, zpt)

    #print((tof * clight.value / 2.) - oneway)
    #print(max(abs((tof * clight.value / 2.) - oneway)))
    #print((tof * clight.value / 2. - oneway)/clight.value*2.)

    # update bouncing point after relativistic correction
    vprj = scpos_tx + zpt * oneway.reshape(-1, 1)
    et_bc = et_tx + oneway/clight.value

    # get planet@bc to bounce point vector
    vbore = vprj - plapos_bc

    # compute off-nadir value and pass/save to df
    offndr = get_offnadir(plapos_bc, scpos_tx, vbore)

    # compute inertial to body-fixed frame rotation
    if XovOpt.get("body") == "MOON":
        # (using np.frompyfunc to vectorize pxform)
        tsipm = pxform_array(vecopts['INERTIALFRAME'], vecopts['PLANETFRAME'], et_bc)
    else: # TODO only works for Mercury!!!!!
        # (using custom implementation)
        # print("tmp_pertPar['dL']", tmp_pertPar['dL'])
        rotpar, upd_rotpar = orient_setup(tmp_pertPar['dRA'], tmp_pertPar['dDEC'], tmp_pertPar['dPM'], tmp_pertPar['dL'])
        tsipm = icrf2pbf(et_bc, upd_rotpar)

    # print(tsipm.ravel)
    # np.stack(tsipm,1)
    # print(tsipm.tolist())
    # print(np.shape(tsipm.tolist()))
    # print(np.shape(vbore))

    # rotate planet@bc to bounce point vector to body fixed frame
    vmbf = [np.dot(tsipm[i], vbore[i]) for i in range(0, np.size(vbore, 0))]
    # print(np.array(vmbf).reshape(-1,3))

    # print(np.linalg.norm(vmbf,axis=1))
    # print(np.linalg.norm(scpos_tx-plapos_bc,axis=1))
    # print(tof)
    # print(oneway)
    # exit()

    # apply tidal deformation (deformation in meters in radial, lon, lat)
    # print("apply tidal corr geoloc_alt")
    dr, dlon, dlat = tidal_deform(vecopts, vmbf, et_bc, SpObj, delta_par=tmp_pertPar)

    # convert xyz to latlon, then apply correction
    rtmp, lattmp, lontmp = astr.cart2sph(np.array(vmbf).reshape(-1, 3))
    # print(rtmp, lattmp, lontmp)
    rtmp += dr
    lattmp += dlat / (vecopts['PLANETRADIUS'] * 1e3)
    lontmp += dlon / (vecopts['PLANETRADIUS'] * 1e3) / np.cos(lattmp)

    # print(dr, dlat / (vecopts['PLANETRADIUS']*1e3) , dlon / (vecopts['PLANETRADIUS']*1e3) / np.cos(lattmp) )
    # exit()

    if (vecopts['OUTPUTTYPE'] == 0):
        vmbf = astr.sph2cart(rtmp, lattmp, lontmp)
        return np.array(vmbf).reshape(-1, 3), et_bc, dr, offndr #2 * oneway / clight.value;
    elif (vecopts['OUTPUTTYPE'] == 1):
        return np.column_stack((np.rad2deg(lontmp), np.rad2deg(lattmp), rtmp)), et_bc, dr, offndr #2 * oneway / clight.value


def get_offnadir(plapos_bc, scpos_tx, vbore):
    vbore_normed = vbore / np.linalg.norm(vbore, axis=1)[:, np.newaxis]
    scxyz_tx = (scpos_tx - plapos_bc)
    scxyz_tx_pbf_normed = np.array(scxyz_tx) / np.linalg.norm(scxyz_tx, axis=1)[:, np.newaxis]
    cos_offndr = np.einsum('ij,ij->i', vbore_normed, scxyz_tx_pbf_normed)
    offndr = np.arccos(np.round(cos_offndr, 10)) # else, when nadir pointing, getting 1.+1.e-12 and throwing warnings
    if np.max(np.abs(offndr)) <= 1:
        offndr = np.rad2deg(offndr)
    else:
        offndr = np.zeros(len(offndr))
    return offndr


def range_corr_iter(Rrx, Rtx, oneway, scpos_rx, scpos_tx, twoway, zpt,itmax=100,tlcbnc = 1.e-3):
    """
    int:type itmax: max number of iterations
    real:type tlcbnc: convergence criteria
    """

    shap_fact = (2 * const.G * const.M_sun / clight ** 2).value
    # avgerr_old = 0
    for it in range(itmax):
        vprj = scpos_tx + zpt * oneway.reshape(-1, 1)
        Rbc = np.linalg.norm(vprj, axis=1)
        Rtx_bc = oneway

        shap_dl = shap_fact * np.log((Rtx + Rbc + Rtx_bc) / (Rtx + Rbc - Rtx_bc))

        rxpt = scpos_rx - vprj
        Rrx_bc = np.linalg.norm(rxpt, axis=1)

        shap_ul = shap_fact * np.log((Rrx + Rbc + Rrx_bc) / (Rrx + Rbc - Rrx_bc))

        avgerr = twoway - (oneway + shap_dl + shap_ul + Rrx_bc)
        # avgerr=shap_dl+shap_ul

        oneway = oneway + 0.5 * avgerr
        # print(it, max(abs(avgerr)))

        if (max(abs(avgerr)) < tlcbnc):
            # if (max(abs(avgerr-avgerr_old))<tlcbnc):
            break
        if (it == itmax - 1):
            print('### geoloc: Max number of iterations reached!')
            print("max resid:", max(abs(avgerr)), "# > tol:", np.count_nonzero(abs(avgerr) > tlcbnc))

        # avgerr_old = avgerr
    return oneway


def get_sc_ssb(et, SpObj, tmp_pertPar, vecopts, t0 = 0):

    # get probe CoM state at TX
    # --------------------------
    if (XovOpt.get("SpInterp") > 0):
        x_sc = np.transpose(SpObj['MGRx'].eval(et))
        v_sc = np.transpose(SpObj['MGRv'].eval(et))
        # print(np.array(x_sc).shape)
        # print(np.array(v_sc).shape)
        scpv = np.concatenate((x_sc, v_sc), axis=1)
        # print(scpv)
        # print(scpv.shape)
        # exit()
    else:
        scpv, lt = spice.spkezr(vecopts['SCNAME'],
                                   et,
                                   vecopts['INERTIALFRAME'],
                                   'NONE',
                                   vecopts['INERTIALCENTER'])
        scpv = np.atleast_2d(np.squeeze(scpv))

    # print('check scpv', scpv)
    # exit()
    scpos = 1.e3 * scpv[:, :3]
    scvel = 1.e3 * scpv[:, 3:]
    # scpos = 1.e3 * np.squeeze(scpv)[:, :3]
    # scvel = 1.e3 * np.squeeze(scpv)[:, 3:]

    # Compute and add ACR offset (if corrections != 0)
    # print([tmp_pertPar[k] for k in ['dA','dC','dR']])
    orb_pert_dict = {k:v for (k,v) in tmp_pertPar.items() for filter_string in ['dA$','dC$','d[A,C,R][0,1,2,c,s]','dR$'] if re.search(filter_string, k)}

    if any(value != 0 for value in orb_pert_dict.values()):
        # print("got in", orb_pert_dict)
        dirs = ['A', 'C', 'R']
        rev_per = 0.5 * 86400 # 12h to secs
        w = 2*np.pi / rev_per
        nvals = len(et)
        dACR = np.zeros((nvals,3))
        for coeff, val in tmp_pertPar.items():
            if coeff in ['dA', 'dC', 'dR'] and val != 0:
                column = dirs.index(coeff[1])
                dACR[:,column] += np.tile(val, len(et))
            elif coeff[:3] in ['dA1', 'dC1', 'dR1'] and val != 0:
                column = dirs.index(coeff[1])
                dACR[:, column] += np.tile(val, len(et)) * sec2day(et-t0)
            elif coeff[:3] in ['dA2', 'dC2', 'dR2'] and val != 0:
                column = dirs.index(coeff[1])
                dACR[:, column] += 0.5 * np.tile(val, len(et)) * np.square(sec2day(et-t0))
            elif coeff[:3] in ['dAc', 'dCc', 'dRc'] and val != 0:
                n_per_orbit = int(coeff[3:])
                column = dirs.index(coeff[1])
                coskwt = np.cos(n_per_orbit * w * (et - t0))
                dACR[:, column] += np.tile(val, len(et))*coskwt
            elif coeff[:3] in ['dAs', 'dCs', 'dRs'] and val != 0:
                n_per_orbit = int(coeff[3:])
                column = dirs.index(coeff[1])
                sinkwt = np.sin(n_per_orbit * w * (et - t0))
                dACR[:, column] += np.tile(val, len(et))*sinkwt

        # from matplotlib import pyplot as plt
        # plt.clf()
        # plt.plot((et - t0),dACRs)
        # plt.savefig('tmp/tst.png')

        # get probe CoM state at TX (w.r.t. planet center)
        # ------------------------------------------------
        scpos_p, scvel_p = get_sc_pla(et, scpos, scvel, SpObj, vecopts)

        dXYZ = astr.rsw_2_xyz(dACR, scpos_p, scvel_p)
        # print("test dACR", dACR, dXYZ)
        # print("test dACR", np.linalg.norm(dACR, axis=1), np.linalg.norm(dXYZ, axis=1))

        # add rotated offset to satpos
        scpos += dXYZ

    return scpos, scvel


def get_sc_pla(et, x_sc, v_sc, SpObj, vecopts):

    if (XovOpt.get("SpInterp") > 0):
        x_pla = np.transpose(SpObj['MERx'].eval(et))
        v_pla = np.transpose(SpObj['MERv'].eval(et))
        scpv_p = np.concatenate((x_sc * 1.e-3 - x_pla, v_sc * 1.e-3 - v_pla), axis=1)
    else:
        scpv_p, lt = spice.spkezr(vecopts['SCNAME'],
                                  et,
                                  vecopts['INERTIALFRAME'],
                                  'NONE',
                                  vecopts['PLANETNAME'])

    scpos_p = 1.e3 * np.array(scpv_p)[:, :3]
    scvel_p = 1.e3 * np.array(scpv_p)[:, 3:]
    return scpos_p, scvel_p

#######################################################################
