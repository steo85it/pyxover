# Options for AccumXov
# from config import XovOpt
import logging

from config import XovOpt


class AccOpt:

    __conf = {
        # env opt
        "sim_altdata": False,
        "get_cov_only": False,

        "remove_max_dist": False,
        "remove_3sigma_median": False,
        "remove_dR200": False,

        "clean_part": True,

        "huber_threshold": 30,
        "distmax_threshold": 0.2,
        "offnad_threshold": 2,
        "h2_limit_on": False,
        # remove worse obs at first iter to speed up
        "downsize": True,  # False #  # False for pawstel
        # extract sample for bootstrap
        "sampling": False,  #
        # rescaling factor for weight matrix, based on average error on xovers at Mercury
        # dimension of meters (to get s0/s dimensionless)
        # useless now, superseeded by adjusting VCE weights
        "sigma_0": 1.,  # 1.e-2 * 2. * 182 # * 0.85 # 0.16 #
        # weights updated by VCE
        "weight_obs": 1.,  # 1.57174113e-06 # 1.
        "weight_constr": 30,  # 2.29140788e-05 # 1.

        "use_advanced_weighting": True,
        # convergence criteria for fixing weights
        "convergence_criteria": 0.05,  # =5%
        # VCE
        "compute_vce": False,  # True

    }
    __setters = list(__conf.keys())

    @staticmethod
    def check_consistency():
        if XovOpt.get('instrument') == "pawstel":
            AccOpt.set("get_cov_only", True)
            AccOpt.set("clean_part", False)
            AccOpt.set("use_advanced_weighting", False)
        # else:
        #     AccOpt.set("get_cov_only", False)
        #     AccOpt.set("clean_part", True)
        #     AccOpt.set("use_advanced_weighting", True)

        logging.info("All good and consistent!")

    @staticmethod
    def get(name):
        return AccOpt.__conf[name]

    @staticmethod
    def set(name, value):
        if name in AccOpt.__setters:
            AccOpt.__conf[name] = value
            print(f"### AccOpt.{name} updated to {value}.")
        else:
            raise NameError("Name not accepted in AccOpt.set() method")