from astropy import units as u
import numpy as np


def sind(x):
    return np.sin(np.deg2rad(x))


def cosd(x):
    return np.cos(np.deg2rad(x))


# Transform vector in rsw orbital frame to inertial
def rsw_2_xyz(vec_in, r_vec, v_vec):
    rot_mat = get_rotmat_xyz_2_rsw(r_vec, v_vec, vec_in)

    # print("vec_in rsw2xyz", vec_in)
    # print("rot_mat rsw2xyz", rot_mat)
    # print("vec_out rsw2xyz", np.einsum('ijk,ij->ik', rot_mat, vec_in))
    # print("vec_out_norm rsw2xyz", np.linalg.norm(vec_in,axis=1))

    # multiply along the right axes (transposed rot_mat)
    return np.einsum('ijk,ij->ik', rot_mat, vec_in)

# Transform vector in inertial to given rsw orbital frame
# TODO weird results
def xyz_2_rsw(vec_in, r_vec, v_vec):
    rot_mat = get_rotmat_xyz_2_rsw(r_vec, v_vec, vec_in)

    # multiply along the right axes (good luck!^^)
    return np.einsum('ijk,ik->ij', rot_mat, vec_in)


def get_rotmat_xyz_2_rsw(r_vec, v_vec, vec_in):
    Rtx = np.linalg.norm(r_vec, axis=1)
    vec_R = (r_vec.T / Rtx).T
    Vtx = np.linalg.norm(v_vec, axis=1)
    vec_A = (v_vec.T / Vtx).T
    vec_C = np.cross(vec_R, vec_A)
    Ctx = np.linalg.norm(vec_C, axis=1)
    vec_C = (vec_C.T / Ctx).T
    # compute third axis (should be close to R for a quasi-circular orbit)
    vec_B = np.cross(vec_A, vec_C)
    Btx = np.linalg.norm(vec_B, axis=1)
    vec_B = (vec_B.T / Btx).T
    # write rotation matrix XYZ -> ACR
    rot_mat = np.concatenate((vec_A, vec_C, vec_B), axis=1).reshape(-1, 3, 3)

    # print("A", vec_A, np.linalg.norm(vec_A))
    # print("C", vec_C, np.linalg.norm(vec_C))
    # print("R", vec_R, np.linalg.norm(vec_R))
    # print("B", vec_B, np.linalg.norm(vec_B))

    # print("vec_in xyz2rsw", vec_in)
    # print("rot_mat xyz2rsw", rot_mat, "det", np.linalg.det(rot_mat))
    # print("vec_out xyz2rsw", np.einsum('ijk,ik->ij', rot_mat, vec_in))
    # print("vec_out_norm xyz2rsw", np.linalg.norm(vec_in, axis=1))

    return rot_mat


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
