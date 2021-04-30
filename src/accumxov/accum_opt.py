# Options for AccumXov
from config import XovOpt

sim_altdata = 0

# print(XovOpt.get("instrument"))

if XovOpt.get("instrument") == "pawstel":
    get_cov_only = True
else:
    get_cov_only = False # True  # True for pawstel

remove_max_dist = False
remove_3sigma_median = False
remove_dR200 = False
# only applied if the above ones are false
if XovOpt.get("instrument") == "pawstel":
    clean_part = False
else:
    clean_part = True

huber_threshold = 30
distmax_threshold = 0.2
offnad_threshold = 2
h2_limit_on = False

# remove worse obs at first iter to speed up
if XovOpt.get("instrument") == "pawstel":
    downsize = False
else:
    downsize = True #False #  # False for pawstel
# extract sample for bootstrap
sampling = False #

# rescaling factor for weight matrix, based on average error on xovers at Mercury
# dimension of meters (to get s0/s dimensionless)
# useless now, superseeded by adjusting VCE weights
sigma_0 = 1. # 1.e-2 * 2. * 182 # * 0.85 # 0.16 #
# weights updated by VCE
weight_obs = 1. # 1.57174113e-06 # 1.
weight_constr = 30 # 2.29140788e-05 # 1.

if XovOpt.get("instrument") == "pawstel":
    use_advanced_weighting = False # for pawstel
else:
    use_advanced_weighting = True

# convergence criteria for fixing weights
convergence_criteria = 0.05 # =5%
#VCE
compute_vce = False # True
