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

import numpy as np
import spiceypy as spice
from astropy import constants as const
from astropy.constants import c as clight

# mylib
import astro_trans as astr
from icrf2pbf import icrf2pbf
from prOpt import SpInterp
from setupROT import setupROT
from tidal_deform import tidal_deform
from prOpt import sim


# from collections import defaultdict


##############################################
# @profile
def geoloc(inp_df, vecopts, tmp_pertPar, SpObj):
    """

    :type inp_df: ladata_df containing TOF(sec) and ET_TX(sec from J2000)
    """
    #  ABCORR = 'NONE'
    # TODO check tof unit (sec or ns???)
    tof = inp_df['TOF'].values
    et_tx = inp_df['ET_TX'].values

    oneway = tof * clight.value / 2.
    twoway = tof * clight.value
    # print('oneway')
    # print(oneway)

    # Set all corrections to 0
    dACR = [0, 0, 0]

    #  print("testObjGeoLoc", SpObj['MGRx'].eval(et_tx))
    #  exit()

    # get probe CoM state at TX
    # --------------------------
    if (SpInterp > 0):
        x_sc = np.transpose(SpObj['MGRx'].eval(et_tx))
        v_sc = np.transpose(SpObj['MGRv'].eval(et_tx))
        # print(np.array(x_sc).shape)
        # print(np.array(v_sc).shape)
        scpv_tx = np.concatenate((x_sc, v_sc), axis=1)
        # print(scpv_tx)
        # print(scpv_tx.shape)
        # exit()
    else:
        scpv_tx, lt = spice.spkezr(vecopts['SCNAME'],
                                   et_tx,
                                   vecopts['INERTIALFRAME'],
                                   'NONE',
                                   vecopts['INERTIALCENTER'])

    scpos_tx = 1.e3 * np.array(scpv_tx)[:, :3]
    scvel_tx = 1.e3 * np.array(scpv_tx)[:, 3:]

    # Compute and add ACR offset (if corrections != 0)
    # print([tmp_pertPar[k] for k in ['dA','dC','dR']])
    if ([tmp_pertPar[k] for k in ['dA', 'dC', 'dR']] != [0, 0, 0]):
        dACR = np.reshape(np.tile([tmp_pertPar[k] for k in ['dA', 'dC', 'dR']], len(et_tx)), (-1, 3))
        # print(dACR.shape)
        # print(len(inp_df))

        # get probe CoM state at TX (w.r.t. planet center)
        # ------------------------------------------------
        if (SpInterp > 0):
            x_pla = np.transpose(SpObj['MERx'].eval(et_tx))
            v_pla = np.transpose(SpObj['MERv'].eval(et_tx))
            # print(np.array(x_sc).shape)
            # print(np.array(v_sc).shape)
            scpv_tx_p = np.concatenate((x_sc - x_pla, v_sc - v_pla), axis=1)
            # print(scpv_tx)
            # print(scpv_tx.shape)
            # exit()
        else:
            scpv_tx_p, lt = spice.spkezr(vecopts['SCNAME'],
                                         et_tx,
                                         vecopts['INERTIALFRAME'],
                                         'NONE',
                                         vecopts['PLANETNAME'])

        scpos_tx_p = 1.e3 * np.array(scpv_tx_p)[:, :3]
        scvel_tx_p = 1.e3 * np.array(scpv_tx_p)[:, 3:]

        dXYZ = astr.rsw_2_xyz(dACR, scpos_tx_p, scvel_tx_p)
        #    print("test dACR", dACR, dXYZ)
        #    print("test dACR", np.linalg.norm(dACR, axis=1), np.linalg.norm(dXYZ, axis=1))

        # add rotated offset to satpos
        scpos_tx += dXYZ
    # update after offset
    Rtx = np.linalg.norm(scpos_tx, axis=1)

    # get probe CoM state at RX
    # --------------------------
    if (SpInterp > 0):
        x_sc = np.transpose(SpObj['MGRx'].eval(et_tx + tof))
        v_sc = np.transpose(SpObj['MGRv'].eval(et_tx + tof))
        # print(np.array(x_sc).shape)
        # print(np.array(v_sc).shape)
        scpv_rx = np.concatenate((x_sc, v_sc), axis=1)
        # print(scpv_tx)
        # print(scpv_tx.shape)
        # exit()
    else:
        scpv_rx, lt = spice.spkezr(vecopts['SCNAME'],
                                   et_tx + tof,
                                   vecopts['INERTIALFRAME'],
                                   'NONE',
                                   vecopts['INERTIALCENTER'])
    scpos_rx = 1.e3 * np.array(scpv_rx)[:, :3]
    scvel_rx = 1.e3 * np.array(scpv_rx)[:, 3:]

    # Compute and add ACR offset (if corrections != 0)
    if ([tmp_pertPar[k] for k in ['dA', 'dC', 'dR']] != [0, 0, 0]):

        # get probe CoM state at RX (w.r.t. planet center)
        # ------------------------------------------------
        if (SpInterp > 0):
            x_pla = np.transpose(SpObj['MERx'].eval(et_tx + tof))
            v_pla = np.transpose(SpObj['MERv'].eval(et_tx + tof))
            # print(np.array(x_sc).shape)
            # print(np.array(v_sc).shape)
            scpv_rx_p = np.concatenate((x_sc - x_pla, v_sc - v_pla), axis=1)
            # print(scpv_tx)
            # print(scpv_tx.shape)
            # exit()
        else:
            scpv_rx_p, lt = spice.spkezr(vecopts['SCNAME'],
                                         et_tx + tof,
                                         vecopts['INERTIALFRAME'],
                                         'NONE',
                                         vecopts['PLANETNAME'])

        scpos_rx_p = 1.e3 * np.array(scpv_rx_p)[:, :3]
        scvel_rx_p = 1.e3 * np.array(scpv_rx_p)[:, 3:]

        dXYZ = astr.rsw_2_xyz(dACR, scpos_rx_p, scvel_rx_p)
        # add rotated offset to satpos
        scpos_rx += dXYZ
    # update after offset
    Rrx = np.linalg.norm(scpos_rx, axis=1)
    # print(scpos_rx)

    # get planet barycenter state (SSB J2000) at bounce
    # --------------------------------------------------
    et_bc = et_tx + tof / 2.

    if (SpInterp > 0):
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

    # get altimeter boresight in S/C frame
    zpt = np.tile(vecopts['ALTIM_BORESIGHT'], np.size(scpos_tx, 0)).reshape(-1, 3)
    # compute s/c frame to inertial rotation (using np.frompyfunc to vectorize pxform)
    # ck+fk+sclk needed

    if (SpInterp > 0):
        quat = SpObj['MGRa'].eval(et_tx)
        quat = np.reshape(np.vstack(np.transpose(quat)), (-1, 4))
        cmat = []
        for i in range(0, len(et_tx)):
            cmat.append(spice.q2m(quat[i, :]))
    else:
        pxform_array = np.frompyfunc(spice.pxform, 3, 1)
        cmat = pxform_array('MSGR_SPACECRAFT', vecopts['INERTIALFRAME'], et_tx)

    # rotate boresight dir to inertial frame
    zpt = [np.dot(cmat[i], zpt[i]) for i in range(0, np.size(zpt, 0))]
    # print(np.array(vmbf).reshape(-1,3))

    if ([tmp_pertPar[k] for k in ['dRl', 'dPt']] != [0, 0]):
        # Apply roll and pitch offsets to zpt
        # print(np.reshape(np.tile([tmp_pertPar[k] for k in ['dRl','dPt']],len(et_tx)),(-1,2)))
        ang_Rl = np.reshape(np.tile([tmp_pertPar[k] for k in ['dRl', 'dPt']], len(et_tx)), (-1, 2))[:, 0]
        ang_Pt = np.reshape(np.tile([tmp_pertPar[k] for k in ['dRl', 'dPt']], len(et_tx)), (-1, 2))[:, 1]

        zpt = astr.rp_2_xyz(zpt, ang_Rl, ang_Pt)
    #    print("test zpt post dRlPt", np.linalg.norm(zpt, axis=1))
    #    print(zpt,np.size(scpos_tx,0))

    # compute corrections to oneway tof - average of Shapiro delay on
    # each branch (!!! Ri, Rj, etc are w.r.t. SSB and not w.r.t. perturbing
    # body, which is wrong but probably acceptable)

    # Max iterations and convergence criteria
    itmax = 0
    tlcbnc = 1.e-3  # meters
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
        #print(it, max(abs(avgerr)))

        if (max(abs(avgerr)) < tlcbnc):
            # if (max(abs(avgerr-avgerr_old))<tlcbnc):
            break
        if (it == itmax - 1):
            print('### geoloc: Max number of iterations reached!')

        # avgerr_old = avgerr

    #print((tof * clight.value / 2.) - oneway)
    #print(max(abs((tof * clight.value / 2.) - oneway)))
    #print((tof * clight.value / 2. - oneway)/clight.value*2.)

    # update bouncing point after relativistic correction
    vprj = scpos_tx + zpt * oneway.reshape(-1, 1)

    # get planet@bc to bounce point vector
    vbore = vprj - plapos_bc

    # compute inertial to body-fixed frame rotation
    if (1 == 2):
        # (using np.frompyfunc to vectorize pxform)
        tsipm = pxform_array(vecopts['INERTIALFRAME'], vecopts['PLANETFRAME'], et_bc)
    else:
        # (using custom implementation)
        rotpar, upd_rotpar = setupROT(tmp_pertPar['dRA'], tmp_pertPar['dDEC'], tmp_pertPar['dPM'], tmp_pertPar['dL'])
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
    dr, dlon, dlat = tidal_deform(vecopts, vmbf, et_bc, SpObj)

    # convert xyz to latlon, then apply correction
    rtmp, lattmp, lontmp = astr.cart2sph(np.array(vmbf).reshape(-1, 3))
    # print(rtmp, lattmp, lontmp)
    rtmp += dr
    lattmp += dlat / (vecopts['PLANETRADIUS'] * 1e3)
    lontmp += dlon / (vecopts['PLANETRADIUS'] * 1e3) / np.cos(lattmp)

    # print(dlat / (vecopts['PLANETRADIUS']*1e3) , dlon / (vecopts['PLANETRADIUS']*1e3) / np.cos(lattmp) )

    # SIM stuff
    if sim:
      rngvec = zpt * oneway.reshape(-1, 1)
      x_pla = np.transpose(SpObj['MERx'].eval(et_tx))
      v_pla = np.transpose(SpObj['MERv'].eval(et_tx))
      scpv_tx_p = np.concatenate((x_sc - x_pla, v_sc - v_pla), axis=1)
      scpos_tx_p = 1.e3 * np.array(scpv_tx_p)[:, :3]

      offndr = np.arccos(np.einsum('ij,ij->i', rngvec, -scpos_tx_p)/
                	 np.linalg.norm(rngvec, axis=1)/
                	 np.linalg.norm(scpos_tx_p, axis=1))
      #print(np.rad2deg(offndr))
      dr = (rtmp-(vecopts['PLANETRADIUS'] * 1e3))*np.cos(offndr)
    
    else:
      dr = 0

    if (vecopts['OUTPUTTYPE'] == 0):
        vmbf = astr.sph2cart(rtmp, lattmp, lontmp)
        return np.array(vmbf).reshape(-1, 3), dr #2 * oneway / clight.value;
    elif (vecopts['OUTPUTTYPE'] == 1):
        return np.column_stack((np.rad2deg(lontmp), np.rad2deg(lattmp), rtmp)), dr #2 * oneway / clight.value

#######################################################################
