"""Coordinate rotation helpers shared across geolocation utilities."""

from astropy import units as u
import numpy as np

from config import XovOpt


def sind(x: np.ndarray | float) -> np.ndarray:
    """Return the sine of an angle in degrees.

    Parameters
    ----------
    x : numpy.ndarray | float
        Angle values expressed in degrees.

    Returns
    -------
    numpy.ndarray
        Sine of the provided angles with the same shape as ``x``.
    """

    return np.sin(np.deg2rad(x))


def cosd(x: np.ndarray | float) -> np.ndarray:
    """Return the cosine of an angle in degrees.

    Parameters
    ----------
    x : numpy.ndarray | float
        Angle values expressed in degrees.

    Returns
    -------
    numpy.ndarray
        Cosine of the provided angles with the same shape as ``x``.
    """

    return np.cos(np.deg2rad(x))


# Transform vector in rsw orbital frame to inertial
def rsw_2_xyz(vec_in: np.ndarray, r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Rotate vectors from the radial–along-track–cross-track (RSW) frame to inertial.

    Parameters
    ----------
    vec_in : numpy.ndarray
        Vectors expressed in the RSW frame with shape ``(N, 3)``.
    r_vec : numpy.ndarray
        Inertial position vectors defining the orbital radius, shape ``(N, 3)``.
    v_vec : numpy.ndarray
        Inertial velocity vectors defining the along-track direction, shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Vectors rotated into the inertial frame with shape ``(N, 3)``.
    """

    rot_mat = get_rotmat_xyz_2_rsw(r_vec, v_vec, vec_in)

    # multiply along the right axes (transposed rot_mat)
    return np.einsum('ijk,ij->ik', rot_mat, vec_in)


# Transform vector in inertial to given rsw orbital frame
# TODO weird results
def xyz_2_rsw(vec_in: np.ndarray, r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Rotate vectors from the inertial frame to the RSW frame.

    Parameters
    ----------
    vec_in : numpy.ndarray
        Vectors expressed in the inertial frame with shape ``(N, 3)``.
    r_vec : numpy.ndarray
        Inertial position vectors defining the orbital radius, shape ``(N, 3)``.
    v_vec : numpy.ndarray
        Inertial velocity vectors defining the along-track direction, shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Vectors rotated into the RSW frame with shape ``(N, 3)``.
    """

    rot_mat = get_rotmat_xyz_2_rsw(r_vec, v_vec, vec_in)

    # multiply along the right axes (good luck!^^)
    return np.einsum('ijk,ik->ij', rot_mat, vec_in)


def get_rotmat_xyz_2_rsw(r_vec: np.ndarray, v_vec: np.ndarray, vec_in: np.ndarray) -> np.ndarray:
    """Compute rotation matrices from inertial coordinates to the RSW frame.

    Parameters
    ----------
    r_vec : numpy.ndarray
        Position vectors defining the radial direction, shape ``(N, 3)``.
    v_vec : numpy.ndarray
        Velocity vectors defining the along-track direction, shape ``(N, 3)``.
    vec_in : numpy.ndarray
        Representative input vectors used only for broadcasting, shape ``(N, 3)``.

    Returns
    -------
    numpy.ndarray
        Rotation matrices with shape ``(N, 3, 3)`` mapping inertial coordinates to RSW.
    """

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

    return rot_mat


def rp_2_xyz(vec_in: np.ndarray, ang_Rl: np.ndarray, ang_Pt: np.ndarray) -> np.ndarray:
    """Apply roll and pitch corrections to rotate input vectors into inertial space.

    Parameters
    ----------
    vec_in : numpy.ndarray
        Vectors to rotate with shape ``(N, 3)``.
    ang_Rl : numpy.ndarray
        Roll angles in radians for each input vector.
    ang_Pt : numpy.ndarray
        Pitch angles in radians for each input vector.

    Returns
    -------
    numpy.ndarray
        Vectors rotated by the provided roll and pitch angles with shape ``(N, 3)``.
    """

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
def cart2sph(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian vectors to spherical coordinates.

    Parameters
    ----------
    xyz : numpy.ndarray
        Input Cartesian coordinates with shape ``(N, 3)`` in meters.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Tuple of radius, latitude, and longitude arrays in meters and radians.
    """

    rtmp = np.linalg.norm(np.array(xyz).reshape(-1, 3), axis=1)
    lattmp = np.arcsin(np.array(xyz).reshape(-1, 3)[:, 2] / rtmp)
    lontmp = np.arctan2(np.array(xyz).reshape(-1, 3)[:, 1], np.array(xyz).reshape(-1, 3)[:, 0])

    return rtmp, lattmp, lontmp


# transform spherical 9 (meters, degrees) to cartesian (meters)
def sph2cart(r: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical coordinates (meters, degrees) to Cartesian meters.

    Parameters
    ----------
    r : numpy.ndarray
        Radius values in meters.
    lat : numpy.ndarray
        Latitude angles in degrees.
    lon : numpy.ndarray
        Longitude angles in degrees.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Cartesian ``x``, ``y``, and ``z`` coordinates in meters.
    """

    x = r * cosd(lon) * cosd(lat)
    y = r * sind(lon) * cosd(lat)
    z = r * sind(lat)

    return x, y, z
