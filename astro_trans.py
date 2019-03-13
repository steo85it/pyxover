from astropy import units as u
import numpy as np


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


# Transform rsw orbital frame to inertial
def rsw_2_xyz(vec_in, r_vec, v_vec):
    Rtx = np.linalg.norm(r_vec, axis=1)
    vec_R = (r_vec.T / Rtx).T
    Vtx = np.linalg.norm(v_vec, axis=1)
    vec_A = (v_vec.T / Vtx).T
    vec_C = np.cross(vec_R, vec_A)
    Ctx = np.linalg.norm(vec_C, axis=1)
    vec_C = (vec_C.T / Ctx).T
    # write rotation matrix ACR -> XYZ
    rot_mat = np.column_stack((vec_A, vec_C, vec_R)).reshape(-1, 3, 3).transpose(0, 2, 1)

    # multiply along the right axes (good luck!^^)
    return np.einsum('ijk,ij->ik', rot_mat, vec_in)


# Transform roll and pitch corrections to inertial
def rp_2_xyz(vec_in, ang_Rl, ang_Pt):
    rot_Rl = np.column_stack(([1] * len(ang_Rl), [0] * len(ang_Rl), [0] * len(ang_Rl),
                              [0] * len(ang_Rl), np.cos(ang_Rl), -np.sin(ang_Rl),
                              [0] * len(ang_Rl), np.sin(ang_Rl), np.cos(ang_Rl))).reshape(-1, 3, 3)

    rot_Pt = np.column_stack((np.cos(ang_Pt), [0] * len(ang_Pt), np.sin(ang_Pt),
                              [0] * len(ang_Pt), [1] * len(ang_Pt), [0] * len(ang_Pt),
                              -np.sin(ang_Pt), [0] * len(ang_Pt), np.cos(ang_Pt))).reshape(-1, 3, 3)

    # Apply pitch and roll offset rotations
    # to the altimeter z dir (in this order)
    tmp = np.einsum('ijk,ij->ik', rot_Pt, vec_in)

    return np.einsum('ijk,ij->ik', rot_Rl, tmp)


# transform cartesian to spherical (meters, radians)
def cart2sph(xyz):
    # print("cart2sph in",np.array(xyz))

    rtmp = np.linalg.norm(np.array(xyz).reshape(-1, 3), axis=1)
    lattmp = np.arcsin(np.array(xyz).reshape(-1, 3)[:, 2] / rtmp)
    lontmp = np.arctan2(np.array(xyz).reshape(-1, 3)[:, 1], np.array(xyz).reshape(-1, 3)[:, 0])

    return rtmp, lattmp, lontmp


# transform spherical 9 (meters, degrees) to cartesian (meters)
def sph2cart(r, lat, lon):
    # print("sph2cart in",r,lat,lon)

    x = r * cosd(lon) * cosd(lat)
    y = r * sind(lon) * cosd(lat)
    z = r * sind(lat)

    return x, y, z
