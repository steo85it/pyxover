import numpy as np
from scipy.optimize import fsolve, root_scalar

def tt2tdb(TT):

    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/time.html#SPICE#20Time#20Representations
    # In SPICE the difference between TT and TDB is computed as follows:
    # 
    #          TDB - TT =  K * sin (E)                  (1)
    # where K is a constant, and E is the eccentric anomaly of the heliocentric orbit of the Earth-Moon barycenter. This difference, which ignores small-period fluctuations, is accurate to about 0.000030 seconds. To five decimal places the difference between TT and TDB is a periodic function with magnitude approximately 0.001658 seconds and period equal to one sidereal year.
    # The eccentric anomaly E is given by
    # 
    #         E = M + EB sin (M)                         (2)
    # where EB and M are the eccentricity and mean anomaly of the heliocentric orbit of the Earth-Moon barycenter. The mean anomaly is in turn given by
    #         M = M0 + M1*t                              (3)
    # where t is the epoch TDB expressed in barycentric dynamical seconds past the epoch of J2000.
    # The values K, EB, M0, and M1 are retrieved from the kernel pool. These are part of the Leapseconds Kernel. They correspond to the ``kernel pool variables'' DELTET/K, DELTET/EB, and DELTET/M. The nominal values are:
    # 
    #    DELTET/K               =    1.657D-3
    #    DELTET/EB              =    1.671D-2
    #    DELTET/M               = (  6.239996D0   1.99096871D-7 )

    # TDB = ET in SPICE
    # TT = TDT of RDRs
    K=1.657e-3
    EB=1.671e-2
    M_=[6.239996e0, 1.99096871e-7]
    # M = M_(1) + M_(2).*TDB;
    # E = M + EB .* sin( M );

    # TT=743235669.031;
    # x = TDB
    def myfunc(x, t):
        return K*np.sin(M_[0]+M_[1]*x + EB*np.sin(M_[0]+M_[1]*x))+t-x

    TDB = []
    for t in TT:
        TDB.append(root_scalar(myfunc,args=(t), bracket=[t-1, t+1]).root)
    
    # find the root of the transcendtal equation: 0 =  K * sin (E) - TDB + TT
    return TDB
