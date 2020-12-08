# Options for AccumXov


sim_altdata = 0


remove_max_dist = False
remove_3sigma_median = False
remove_dR200 = False
# only applied if the above ones are false
clean_part = True
huber_threshold = 30
distmax_threshold = 0.2
offnad_threshold = 2
h2_limit_on = False

# remove worse obs at first iter to speed up
downsize = True #False #
# extract sample for bootstrap
sampling = False #

# rescaling factor for weight matrix, based on average error on xovers at Mercury
# dimension of meters (to get s0/s dimensionless)
# useless now, superseeded by adjusting VCE weights
sigma_0 = 1. # 1.e-2 * 2. * 182 # * 0.85 # 0.16 #
# weights updated by VCE
weight_obs = 1. # 1.57174113e-06 # 1.
weight_constr = 30 # 2.29140788e-05 # 1.
# convergence criteria for fixing weights
convergence_criteria = 0.05 # =5%
#VCE
compute_vce = False # True
