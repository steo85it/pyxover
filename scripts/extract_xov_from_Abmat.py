
#!/usr/bin/env python3
# # from xovutils.astrotrans import rsw_2_xyz # en vectoriel
import os
from  wd_utils import xov_header_cvs, xov_table_cvs
import time
import pickle
import csv

id = "DA0"
iter = 0
suffix = "_nosol"
#suffix = "_A"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
out_folder = f"{data_path}OBS/"

file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
              f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl",
              f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_G.pkl"]

# id = f"{id}_l70"
# id = f"{id}_nodownsize"
# id = f"{id}_semnodownsize"


startInit = time.time()
csv_filename = f"{out_folder}xov_{id}_{iter}{suffix}_Abmat.csv"

with open(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}{suffix}.pkl", "rb") as file:
   import scipy as sp
   Abmat = pickle.load(file)
   xov = Abmat.xov
   # sp.sparse.save_npz(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}_pen.npz", Abmat.penalty_mat)
   # sp.sparse.save_npz(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}_penavg.npz", Abmat.penalty_mat_avg)

file = open(csv_filename,'w')
xov_header_cvs(file, xov_time=False,  distance=False, derivatives=False)

writer = csv.writer(file, delimiter='\t')

contain_nan = xov_table_cvs(xov, writer, xov_time=False,  distance=False, derivatives=False)
if contain_nan:
   print(f"Found nan in Abmat*.pkl")   
      
endInit = time.time()
file.close()
endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")