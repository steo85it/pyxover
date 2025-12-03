import pickle
import time
import gc
import pandas as pd
import numpy as np
import itertools as itert
import matplotlib.pyplot as plt


import sys
import os.path
import glob
import csv


fig_name = f"orbcorr_{id}_it0-3"
fig_name = f"{id}_all"

plot_orbcorr  = False # plot correction (solution)
plot_error    = False # plot orbit error
plot_globcorr = True
std_sol = True # formal error or solution
export_cvs = False


data_path = "/storage/research/aiub_gravdet/WD_BELA/"
in_folder = f"{data_path}pyXover/out/{id}_0/"

sol = []
std = []
glob_nam = ['RA', 'DEC','PM','L', 'h2']
glob_nam = ['RA', 'DEC','PM','L']
glob_plt = glob_nam
# glob_nam = ['RA', 'DEC','PM', 'h2']
# glob_nam += [f'LIB{i}' for i in range(1, 12)]
# glob_plt = ['RA', 'DEC','PM', 'h2']
# glob_plt = [f'LIB{i}' for i in range(1, 6)]
# glob_plt = [f'LIB{i}' for i in range(6, 12)]
plot_bertone2021 = False

glob_lbl = {'RA' : r"$\alpha_0$ [as]",
            'DEC': r"$\delta_0$ [as]",
            'PM' : r"$\omega$ [as.yr$^{-1}$]",
            'L'  : r"$L$ [as]",
            'h2' : r"$h_2$ [-]"}
for i in range(1, 12):
    glob_lbl[f'LIB{i}'] = fr"$\lambda_{{{i}}}$ [as]"


# Only Northern hemisphere
# fid_ = ["CA0", "AB0", "AB0_CA0", "BA1", "AB0_BA1", "AB0_BA1_CA0"]
# leg = ["MLA", "BELA_N", "BELA_N+MLA", "BELA_N/MLA", "BELA_N+\nBELA_N/MLA", "BELA_N+MLA+\nBELA_N/MLA"]
#fid_ = ["CA0", "AB0", "AB0_CA0", "BA1", "AB0_BA1", "AB0_BA1_CA0"]
#leg = ["MLA", "BELA_N", "BELA_N+\nMLA", "BELA_N/MLA", "BELA_N+\nBELA_N/MLA","BELA_N+MLA+\nBELA_N/MLA"]

#fid_ = ["CA0", "AB0", "AB0_AB1", "BA1"]
#leg = ["MLA", "BELA_N", "BELA", "BELA_N/MLA"]

#fid_ = ["CA0", "AB0_CA0", "AB0_AB1_CA0", "AB0_AB1_BA1_CA0"]
#leg = ["MLA", "BELA_N+\nMLA", "BELA+MLA", "BELA+MLA+\nBELA_N/MLA", ]

# all with BELA
# fid_ = ["AB0_AB1", "AB0_AB1_CA0", "AB0_AB1_BA1", "AB0_AB1_BA1_CA0"]
# leg = ["BELA", "BELA+MLA", "BELA+\nBELA/MLA", "BELA+MLA+\nBELA/MLA"]


time1 = time.perf_counter()
file_names = []

id = "CA1"
iter = 0
file_names = [f"{data_path}pyXover/out/{id}_0/Abmat_{id}_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/{id}_0/Abmat_{id}_q3_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/{id}_0/Abmat_{id}_nodownsize_{iter}_{iter+1}.pkl"]
file_names = [f"{data_path}pyXover/out/{id}_0/Abmat_{id}_l70_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/{id}_0/Abmat_{id}_nodownsize_{iter}_{iter+1}.pkl"]
leg = ["q=0.1", "q=0.3", "no downsize"]
id = "BA3"
subid_ = ["l70","l80","l80v2", "nodownsize"]
subid_ = ["l80","l80v2"]
subid_ = ["l80v3"]
# id = "AB2"
# subid_ = ["l80", "seml80"]
# subid_ = ["Nl80", "seml80"]
# id = "CA1"
# subid_ = ["l70"]
leg = subid_



file_names = []
for subid in subid_:
   # file_names.append(f"{data_path}pyXover/out/{id}_0/Abmat_{id}_all_{subid}_{iter}_{iter+1}.pkl")
   file_names.append(f"{data_path}pyXover/out/{id}_0/Abmat_{id}_{subid}_{iter}_{iter+1}.pkl")
   
file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nl80_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/BA3_0/Abmat_BA3_all_l80v3_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/BA3_0/Abmat_BA3_allN_l80_{iter}_{iter+1}.pkl"]
file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nnodownsize_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/BA3_0/Abmat_BA3_allN_nodownsize_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/BA3_0/Abmat_BA3_nodownsizev2_{iter}_{iter+1}.pkl"]
file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nnodownsize_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nl80_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/AB2_0/Abmat_AB2_l80_{iter}_{iter+1}.pkl"]
leg = ["BELA_N","COMB BELA_N","COMB BELA_N_2"]
leg = ["semi, nodownsize", "semi, l=80°", "full, l=80°"]
# file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_l80_{iter}_{iter+1}.pkl",
#               f"{data_path}pyXover/out/AB2_0/Abmat_AB2_nodownsize_{iter}_{iter+1}.pkl"]
# leg = ["l=80°", "no downsize"]
# p_title = "All full tracks, BELA only"

# file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nl80_{iter}_{iter+1}.pkl",
#               f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nnodownsize_{iter}_{iter+1}.pkl"]
# file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nl80alllowlat_{iter}_{iter+1}.pkl",
#               f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nnodownsize_{iter}_{iter+1}.pkl"]
# leg = ["l=80°", "no downsize"]
# p_title = "Northern hemisphere, semi tracks, BELA only"
file_names = [f"{data_path}pyXover/out/AB2_0/Abmat_AB2_Nnodownsize_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/AB2_0/Abmat_AB2_seml80_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/AB2_0/Abmat_AB2_semnodownsize_{iter}_{iter+1}.pkl"]
file_names = [f"{data_path}pyXover/out/CA1_0/Abmat_CA1_nodownsize_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/AB2_0/Abmat_AB2_semnodownsize_{iter}_{iter+1}.pkl",
              f"{data_path}pyXover/out/BA3_0/Abmat_BA3_all_nodownsizev2_{iter}_{iter+1}.pkl"]
leg = ["MLA only", "BELA only", "combined"]
p_title = "All semi tracks from BELA"
p_title = "All semi tracks from MLA"
file_names = [f"{data_path}pyXover/out/CA5_0/Abmat_CA5_0_1.pkl",
              f"{data_path}pyXover/out/BA4_0/Abmat_BA4_0_1.pkl"]
#file_names = [f"{data_path}pyXover/out/CA4_0/Abmat_CA4_0_1.pkl",
#              f"{data_path}pyXover/out/CA4_3/Abmat_CA4_3_4.pkl"]
# leg = ["perturbed tracks"]
# p_title = "MLA only"
leg = ["MLA only","combined"]
# file_names = [f"{data_path}pyXover/out/AC0_0/Abmat_AC0_{iter}_{iter+1}.pkl"]
# leg = range(0,2)
orb = '1'
# p_title = "BELA only"
# file_names = [f"{data_path}pyXover/out/AC0_0/Abmat_AC0_{iter}_{iter+1}.pkl",
#               f"{data_path}pyXover/out/AC0_1/Abmat_AC0_{iter+1}_{iter+2}.pkl",
#               f"{data_path}pyXover/out/AC0_2/Abmat_AC0_{iter+2}_{iter+3}.pkl"]
file_names = [f"{data_path}pyXover/out/AC4_0/Abmat_AC4_0_1.pkl",
              f"{data_path}pyXover/out/BA4_0/Abmat_BA4_0_1.pkl"]
leg = ["BELA only", "combined"]
# leg = range(0,3)
orb = '1'
file_names = [f"{data_path}pyXover/out/CA5_0/Abmat_CA5_0_1.pkl",
              f"{data_path}pyXover/out/AC4_0/Abmat_AC4_0_1.pkl",
              f"{data_path}pyXover/out/BA4_0/Abmat_BA4_0_1.pkl"]
file_names = [f"{data_path}pyXover/out/AC1_0/Abmat_AC1_0_1.pkl",
              f"{data_path}pyXover/out/AC4_0/Abmat_AC4_0_1.pkl",
              f"{data_path}pyXover/out/BA4_0/Abmat_BA4_0_1.pkl"]
leg = ["MLA only", "BELA only", "combined"]
leg = ["BELA_S only", "combined_S"]

file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1.pkl"]
leg = ["MLA_CB0", "MLA_BA7"]
# file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1.pkl",
#                f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1.pkl",
#                f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1.pkl"]
# leg = ["MLA_CB0", "MLA_CB1","BELA_AD2"]
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_extended.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_extended.pkl"]
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1.pkl"]
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_I.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_B.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_J.pkl"]
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_I.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_J.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_nominal.pkl"]
# file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_extended.pkl"]
leg = ["BELA_AD2", "BELA_BA7", "BELA_BA7", "BELA_BA7"]
leg = [0,1,2,3]
p_title = "BELA only, all tracks"
# file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1.pkl",
#               f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1.pkl"]
# file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1.pkl",
#               f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_extended.pkl"]
# p_title = "MLA only"
# file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1.pkl",
#               f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1.pkl",
#               f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_I.pkl",
#               f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1.pkl",
#               f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_extended.pkl"]
file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_K.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_nominal.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl"]
file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_A.pkl",
              f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_B.pkl",
              f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_C.pkl"]
leg = ["A","B","C"]

orb = '2'
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_nominal.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_I.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_J.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_K.pkl"]
leg = ["nom","I","J","K"]
file_names = [f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_nominal.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_B.pkl"]
leg = ["nom","A","B"]
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_K.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl"]
leg = ["BELA_only","BELA_combined"]
# file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_B.pkl",
#               f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_K.pkl",
#               f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl"]
# leg = ["MLA","BELA","BELA/MLA"]
orb = '1'
# file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_B.pkl",
#               f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl"]
# leg = ["MLA_only","MLA_combined"]
file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_B.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_K.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_L.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_C.pkl"]
leg = ["MLA","BELA","BELA_ext","BELA/MLA", "BELA/MLA_ext"]
file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_B.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_C.pkl"]
leg = ["MLA","BELA/MLA", "BELA/MLA_ext"]
file_names = [f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_K.pkl",
              f"{data_path}pyXover/out/AD2_0/Abmat_AD2_0_1_L.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_A.pkl",
              f"{data_path}pyXover/out/BA7_0/Abmat_BA7_0_1_C.pkl"]
leg = ["BELA","BELA_ext","BELA/MLA", "BELA/MLA_ext"]
file_names = [f"{data_path}pyXover/out/CB0_0/Abmat_CB0_0_1.pkl",
              f"{data_path}pyXover/out/BA8_0/Abmat_BA8_0_1_B.pkl"]
leg = ["MLA_only","MLA_combined"]
file_names = [f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_A.pkl",
              f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_C.pkl",
              f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_D.pkl"]
file_names = [f"{data_path}pyXover/out/CB0_0/Abmat_CB0_0_1.pkl",
              f"{data_path}pyXover/out/CB2_0/Abmat_CB2_0_1_A.pkl",
              f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_A.pkl"]
leg = ["A","B","C"]
file_names = [f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_A.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_B.pkl",
              f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_B.pkl"]
file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_A.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_B.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_B.pkl"]
file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_A.pkl",
              # f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_B.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_B.pkl"]
file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl"]
file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_G.pkl"]
# pointing error extended
file_names = [f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_B.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
              f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_C.pkl"]
# pointing error nominal
file_names = [f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_B.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
              f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_D.pkl"]
p_title = "BELA only, all tracks"
leg = ["MLA","BELA","BELA/MLA"]
# nominal: (no) pointing error
file_names = [f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_B.pkl",
              f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
              f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_D.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl"]
p_title = "BELA only, all tracks"
leg = ["MLA$_p$","MLA","BELA","BELA/MLA$_p$","BELA/MLA"]

# MLA tracks nominal: pointing error
orb = '1'
file_names = [f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_B.pkl",
              f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_D.pkl",]
p_title = "MLA only, with pointing errors"
# MLA tracks nominal: pointing error
file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl",]
p_title = "MLA only, without pointing errors"

# BELA tracks nominal: pointing error
# orb = '2'
# file_names = [f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
#               f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_D.pkl",]
# p_title = "BELA only, all tracks"

# no pointing nominal/extended mission
file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_G.pkl"]
orb = '1'
#file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
#              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl"]
# leg = ["MLA only", "MLA/BELA"]
# file_names = [f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
              # f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl",
#              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_G.pkl"]
#orb = '2'
leg = ["BELA","BELA_ext","MLA/BELA", "MLA/BELA_ext"]
# leg = ["BELA","MLA/BELA","BELA$_{ext}$"]
# pointing error nominal/extended mission
# file_names = [f"{data_path}pyXover/out/CB3_0/Abmat_CB3_0_1_B.pkl",
#               f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
#               f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
#               f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_D.pkl",
#               f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_C.pkl"]
leg = ["MLA","BELA","BELA$_{ext}$","MLA/BELA","MLA/BELA$_{ext}$"]
# orb = '2'
# file_names = [f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_B.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_B.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_A.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_C.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_D.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_E.pkl"]
# leg = ["BELA","BELA/MLA w=1","BELA/MLA w=0.7","BELA/MLA w=0.5","BELA/MLA w=0.25","BELA/MLA w=0.01"]

              
# leg = ["MLA","BELA"]
# file_names = [f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_B.pkl",
#               f"{data_path}pyXover/out/BA9_0/Abmat_BA9_0_1_B.pkl"]

file_names = [f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_B.pkl",
              f"{data_path}pyXover/out/AD6_0/Abmat_AD6_0_1_B.pkl",
              f"{data_path}pyXover/out/BB1_0/Abmat_BB1_0_1_B.pkl"]
leg = ["MLA","BELA$_{ext}$","MLA/BELA$_{ext}$"]
file_names = [f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_B.pkl",
              f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_C.pkl",
              f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_D.pkl",
              f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_E.pkl"]
leg = ["[1,1,0]","[0.003,70.7,0]","[0.0004, 128, 0.41]","[0.003, 77.4, 1e8]"]
file_names = [f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_B.pkl",
              f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_C.pkl",
              f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_D.pkl"]
leg = ["[1,1,0]","[0.003,70.7,0]","[0.0004, 128, 0.41]"]

orb = '2'
file_names = [f"{data_path}pyXover/out/AD6_0/Abmat_AD6_0_1_B.pkl",
              f"{data_path}pyXover/out/AD6_0/Abmat_AD6_0_1_C.pkl"]
leg = ["[1,1,0]","[0.002,0.349,0]"]

orb = '1'
file_names = [f"{data_path}pyXover/out/CB5_0/Abmat_CB5_0_1_G.pkl",
              f"{data_path}pyXover/out/AD6_0/Abmat_AD6_0_1_E.pkl",
              f"{data_path}pyXover/out/BB1_0/Abmat_BB1_0_1_F.pkl"]
file_names = [f"{data_path}pyXover/out/CB6_0/Abmat_CB6_0_1_E.pkl",
              f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_B.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_B.pkl"]
leg = ["MLA","BELA$_{ext}$","MLA/BELA$_{ext}$"]
file_names = [f"{data_path}pyXover/out/CB6_0/Abmat_CB6_0_1_E.pkl",
              f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_F.pkl",
              f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_H.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_F.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_G.pkl"]
leg = ["MLA","BELA$_{nom}$","BELA$_{nom}$ tr","MLA/BELA$_{nom}$","MLA/BELA$_{nom}$ tr"]

orb = '1'
file_names = [f"{data_path}pyXover/out/CB6_0/Abmat_CB6_0_1_E.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_F.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_G.pkl"]
leg = ["MLA","MLA/BELA$_{nom}$","MLA/BELA$_{nom}$ tr"]
file_names = [f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_F.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_G.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_I.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_H.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_J.pkl"]
leg = ["no threshold","$|\phi_{all}|<85$","$|\phi|<80$,$|\phi_{BB}|<88$",
       "$|\phi|<85$,$|\phi_{M/B}|<75$","$|\phi|<75$,$|\phi_{BB}|<85$"]

file_names = [f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_F.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_H.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_J0.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_I.pkl",
              f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_N.pkl"]
leg = ["no threshold","$|\phi|<85$,$|\phi_{M/B}|<75$","$|\phi|<75$,$|\phi_{BB}|<85$",
       "bands, nmin = 300","bands, nmin = 1000"]

# orb = '2'
# file_names = [f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_F.pkl",
#               f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_H.pkl",
#               f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_F.pkl",
#               f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_G.pkl"]
# leg = ["BELA$_{nom}$","BELA$_{nom}$ tr","MLA/BELA$_{nom}$","MLA/BELA$_{nom}$ tr"]


# file_names = [f"{data_path}pyXover/out/CB6_0/Abmat_CB6_0_1_E.pkl",
#               f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_B.pkl"]
# leg = ["MLA","MLA/BELA$_{ext}$"]
# orb = '2'
# file_names = [f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_B.pkl",
#               f"{data_path}pyXover/out/BB2_0/Abmat_BB2_0_1_B.pkl"]
# leg = ["BELA$_{ext}$","MLA/BELA$_{ext}$"]
# leg = ["MLA","MLA/BELA$_{ext}$"]

orb = '2'
file_names = [f"{data_path}pyXover/out/AE0_0/Abmat_AE0_0_1_A.pkl"]
leg = ["BELA$_{ext}$"]

file_names = [f"{data_path}pyXover/out/CB9_0/Abmat_CB9_0_1_A.pkl",
              f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_A.pkl",
              f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_E.pkl"]
leg = ["MLA", "BELA", "MLA/BELA"]
file_names = [f"{data_path}pyXover/out/CB9_0/Abmat_CB9_0_1_A.pkl",
              f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_C.pkl",
              f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_A.pkl",
              f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_B.pkl",
              f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_E.pkl"]
leg = ["MLA", "BELA", "BELA$_{ext}$", "MLA+BELA","MLA+BELA$_{ext}$"]
color = ['#000000','#4477AA', '#228833',
            '#66CCEE', '#EE6677', '#CCBB44']
# file_names = [f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_C.pkl",
#               f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_I.pkl",
#               f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_J.pkl",
#               f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_K.pkl",
#               f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_F.pkl",]
# leg = ["all obs.", "80% hilat", "50% hilat", "10% hilat", "2 blocks"]
# color = ['#4477AA', '#228833', '#66CCEE', '#EE6677', '#CCBB44']

# file_names = [f"{data_path}pyXover/out/CB9_0/Abmat_CB9_0_1_A.pkl",
#               f"{data_path}pyXover/out/CC4_0/Abmat_CC4_0_1_A.pkl",
#               f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_A.pkl",
#               f"{data_path}pyXover/out/AE4_0/Abmat_AE4_0_1_A.pkl"]
# file_names = [f"{data_path}pyXover/out/CC4_0/Abmat_CC4_0_1_A.pkl",
#               f"{data_path}pyXover/out/AE4_0/Abmat_AE4_0_1_B.pkl",
#               f"{data_path}pyXover/out/CC4_0/Abmat_BB4_0_1_A.pkl",
#               f"{data_path}pyXover/out/BB4_0/Abmat_BB4_0_1_A.pkl"]
# leg = ["MLA", "BELA", "MLA&BELA", "MLA+BELA"]
# color = ['#4477AA', '#228833', '#66CCEE', '#EE6677', '#CCBB44']
# file_names = [f"{data_path}pyXover/out/CB9_0/Abmat_CB9_0_1_A.pkl",
#               f"{data_path}pyXover/out/CC4_0/Abmat_CC4_0_1_A.pkl",
#               f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_A.pkl",
#               f"{data_path}pyXover/out/AE4_0/Abmat_AE4_0_1_B.pkl",
#               f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_E.pkl",
#               f"{data_path}pyXover/out/BB4_0/Abmat_BB4_0_1_A.pkl"]
# leg = ["MLA L", "MLA LIB", "BELA L", "BELA LIB","MLA+BELA L", "MLA+BELA LIB"]
# orb = '1'
# file_names = [f"{data_path}pyXover/out/CB9_old_0/Abmat_CB9_0_1_A.pkl",
#               f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_B.pkl",
#               f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_E.pkl"]
# leg = ["MLA", "MLA+BELA", "MLA+BELA$_{ext}$"]
# color = ['#4477AA', '#EE6677', '#CCBB44']
# orb = '2'
# color = ['#228833','#EE6677']
# file_names = [f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_C.pkl",
#               f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_B.pkl"]
# leg = ["BELA", "MLA+BELA"]
# file_names = [f"{data_path}pyXover/out/AE2_0/Abmat_AE2_0_1_A.pkl",
#               f"{data_path}pyXover/out/BB3_0/Abmat_BB3_0_1_E.pkl"]
# leg = ["BELA$_{ext}$", "MLA+BELA$_{ext}$"]
# color = ['#66CCEE','#CCBB44']

# file_names = [f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_A.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_B.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_C.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_D.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_E.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_F.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_G.pkl",
#               f"{data_path}pyXover/out/CC5_0/Abmat_CC5_0_1_H.pkl",]



fig_name = f"{id}_all"
fig_name = f"{id}"

xlims = [20,60,3]


if std_sol:
   fig_name = f"std_{fig_name}"
else:
   fig_name = f"corr_{fig_name}"

if plot_orbcorr:
   fig_name = f"orb{fig_name}"
else:
   fig_name = f"glob{fig_name}"

for file_name in file_names:
   if os.path.isfile(file_name):
      with open(file_name, "rb") as f:
         Amat = pickle.load(f)
         print(f"m0={Amat.resid_wrmse}")
         Amat.sol_dict['std'].update((x, y*Amat.resid_wrmse) for x, y in Amat.sol_dict['std'].items())
         print("vce_obs", Amat.vce_obs)
         print("vce_pen", Amat.vce_pen)
         std.append(Amat.sol_dict['std'])
         sol.append(Amat.sol_dict['sol'])
   else:
      print("File does not exist", file_name)

if plot_orbcorr and plot_error:
   if orb == '1':
      id_ref = "CB9"
      #id_ref = "AD2"
      allFiles = glob.glob(f"{data_path}pyXover/out/{id_ref}_0/gtrack_*/gtrack_*.pkl")
   else:
      id_ref = ["AE2","AE3"]
      allFiles = glob.glob(f"{data_path}pyXover/out/{id_ref[0]}_0/gtrack_*/gtrack_*.pkl") + \
         glob.glob(f"{data_path}pyXover/out/{id_ref[1]}_0/gtrack_*/gtrack_*.pkl")
   
   dA = dict()
   dC = dict()
   dR = dict()
   for file_name in allFiles:
      with open(file_name, "rb") as f:
         track = pickle.load(f)
         track_name = str(track.name)
         dA[track_name] = track.pert_cloop_0['dA']
         dC[track_name] = track.pert_cloop_0['dC']
         dR[track_name] = track.pert_cloop_0['dR']
# print(sol)
sol_ref = sol[0]
# sol = sol[1:]
if std_sol:
   sol = std
if plot_orbcorr:
   xmax = 0.5
   fig, axs = plt.subplots(3,figsize=(6.1,4.8))
   l = []
   for arg,labl,col in zip(sol,leg,color):
      stdA = dict()
      stdC = dict()
      stdR = dict()
      for par in arg:
      # for par in sol_ref:
         if not (par in sol_ref):
            continue
         if par.startswith(orb):
            track_name = par.split('_')[0]
            if plot_error and not track_name in dA.keys():
               continue
            if par.endswith('A') or par.endswith('C') or par.endswith('R'):
               if arg[par] == 0:
                  print("Large correction removed")
                  continue
            if par.endswith('A'):
               stdA[track_name] = arg[par]
            elif par.endswith('C'):
               stdC[track_name] = arg[par]
            elif par.endswith('R'):
               stdR[track_name] = arg[par]
                  
      nbins = 100
      if plot_error:
         stdA = { track : stdA[track]+dA[track] for track in stdA.keys() }
         stdC = { track : stdC[track]+dC[track] for track in stdC.keys() }
         stdR = { track : stdR[track]+dR[track] for track in stdR.keys() }
      # print(len(stdA))
      # print([s for s in stdA if s==0])
      sub_tracks = [int(track) for track in stdR.keys() if np.abs(stdR[track])<1e-2]
      sub_tracks.sort()
      print(sub_tracks[:10])
      stdA = list(stdA.values())
      stdC = list(stdC.values())
      stdR = list(stdR.values())
      
      stdA = [s for s in stdA if s<xlims[0]]
      stdC = [s for s in stdC if s<xlims[1]]
      stdR = [s for s in stdR if s<xlims[2]]
      
      legi = [labl, None, None]
      std_ACR = [np.std(stdA), np.std(stdC), np.std(stdR)]
      mean_ACR = [np.mean(stdA), np.mean(stdC), np.mean(stdR)]
      print("std in ACR", std_ACR)
      print("mean in ACR",mean_ACR)
      #legi = [fr"$m$={mean_ACR[0]:.1f}, $\sigma$={std_ACR[0]:.1f}",f"m={mean_ACR[1]:.1f}, s={std_ACR[1]:.1f}",f"m={mean_ACR[2]:.1f}, s={std_ACR[2]:.1f}"]
      legi = [fr"$m$={m:.1f}, $\sigma$={s:.1f}" for m,s in zip(mean_ACR,std_ACR)]
      
      

      axs[0].hist(stdA,nbins, alpha = 0.5, label=legi[0],color=col)
      axs[0].legend(loc="upper right")
      axs[1].hist(stdC,nbins, alpha = 0.5, label=legi[1],color=col)
      axs[1].legend(loc="upper right")
      axs[2].hist(stdR,nbins, alpha = 0.5, label=legi[2],color=col)
    
      
      if export_cvs and len(stdA)>0:
         csv_filename = f"{data_path}pyXover/orbcorr_{labl}.csv"

         file = open(csv_filename,'w')
         writer = csv.writer(file, delimiter='\t')
         writer.writerow(stdA)
         writer.writerow(stdC)
         writer.writerow(stdR)
         file.close()
   
   # axs[0].hist(Amat.xov.pert_cloop.dA,nbins, alpha = 0.5)
   # axs[1].hist(Amat.xov.pert_cloop.dC,nbins, alpha = 0.5)
   # axs[2].hist(Amat.xov.pert_cloop.dR,nbins, alpha = 0.5)
   axs[0].set_ylabel('Along-track')
   axs[1].set_ylabel('Cross-track')
   axs[2].set_ylabel('Radial')
   axs[0].set_xlim([-xlims[0], xlims[0]])
   axs[1].set_xlim([-xlims[1], xlims[1]])
   axs[2].set_xlim([-xlims[2], xlims[2]])
   if orb == '1':
      axs[2].set_xlabel('MLA track error [m]')
   else:
      axs[2].set_xlabel('BELA track error [m]')
   # for ax in axs:
      # ax.set_ylabel('count')
      # ax.set_xlim([0, xmax])
   plt.legend()
   handles, labels = plt.gca().get_legend_handles_labels()
   fig.legend(handles, leg, loc='upper center', ncols = 3)
   # fig.suptitle(p_title)

   plt.savefig(f"{fig_name}.png")
   plt.savefig(f"{fig_name}.svg")
   plt.savefig(f"{fig_name}.pdf")

if plot_globcorr:
   # color = ['k','b','g','r','c','m']
   # color = ['#000000','#4477AA', '#EE6677', '#228833', '#CCBB44',
   #          '#66CCEE','#AA3377', '#BBBBBB']
   if export_cvs:
      i=-1
      for name in glob_nam:
         i+=1
         par = f"dR/d{name}"
         height = []
         height_std = []
         for arg in sol:
            height.append(abs(arg[par]))
         for arg in std:
            height_std.append(abs(arg[par]))
           
         csv_filename = f"{data_path}pyXover/globsol_extended_{name}.csv"
         file = open(csv_filename,'w')
         writer = csv.writer(file, delimiter='\t')
         writer.writerow(leg)
         writer.writerow(height)
         writer.writerow(height_std)
         file.close()
   
   sol_dict = dict()
   std_dict = dict()
   if plot_bertone2021:
      sol_dict['Bertone+2021'] = [1e-5,1e-5,1e-5,1e-5,1e-5]
      std_dict['Bertone+2021'] = [5.4e-5*3600/3, 2.8e-5*3600/3, 1.5e-7*3600*365.25/3, 0.2/3, 0.3/3]
   # sol_dict = std_dict # WD: just show the formal errors
   for name, arg in zip(leg,sol):
      sol_dict[name] = [abs(arg[f"dR/d{par_name}"]) for par_name in glob_plt]
   for name, arg in zip(leg,std):
      std_dict[name] = [abs(arg[f"dR/d{par_name}"]) for par_name in glob_plt]

   x = np.arange(len(glob_plt))  # the label locations
   width = 0.25  # the width of the bars
   width = 0.15  # the width of the bars

   fig, ax = plt.subplots(layout='constrained',figsize=(8,4.8))

   multiplier = 0
   for attribute, measurement in sol_dict.items():
      offset = width * (multiplier-1.5)
      rects = ax.bar(x + offset, measurement, width, label=attribute, facecolor=color[multiplier], edgecolor=None, alpha=0.4)
      # ax.bar_label(rects, padding=3)
      multiplier += 1

   ax.legend(loc='upper left', ncols=np.ceil(len(leg)/2))
   
      
   multiplier = 0
   for attribute, measurement in std_dict.items():
      offset = width * (multiplier-1.5)
      rects = ax.bar(x + offset, measurement, width, label=attribute, fill=False, edgecolor=color[multiplier], linewidth = 2)
      # ax.bar_label(rects, padding=3)
      multiplier += 1

   # Add some text for labels, title and custom x-axis tick labels, etc.
   name2 = [glob_lbl[n] for n in glob_plt]
   ax.set_xticks(x + width, name2)
   ax.set_ylim(1e-4, 2e-1)
   # ax.set_ylim(1e-3, 1)
   # ax.set_ylim(2e-3, 3e-2)
   
   # plt.xticks(leg, leg, rotation='vertical')
   plt.yscale("log")

   # ax.set_ylabel('Formal and true errors')
   ax.set_ylabel('Formal errors')
   dir = "/storage/homefs/desprats/pyxover/plot/examples/BELA/"
   plt.savefig(f"{fig_name}.png")
   plt.savefig(f"{fig_name}.svg")
   plt.savefig(f"{fig_name}.pdf")


   
time2 = time.perf_counter()
print("Elapsed time: ", time2-time1)