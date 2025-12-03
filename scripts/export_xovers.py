
#!/usr/bin/env python3
# # from xovutils.astrotrans import rsw_2_xyz # en vectoriel
import os
from  wd_utils import xov_pkl2csv, getgtrackt0, xov_pkl2csv2
import time

id1 = "CB4_0"
id2 = "CB4_0"
# file_names = [f"{data_path}pyXover/out/CB4_0/Abmat_CB4_0_1_C.pkl",
#               f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_C.pkl",
#               f"{data_path}pyXover/out/AD4_0/Abmat_AD4_0_1_D.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_F.pkl",
#               f"{data_path}pyXover/out/BB0_0/Abmat_BB0_0_1_G.pkl"]
# data_path = "/storage/homefs/desprats/pyxover/examples/BELA/data/"
# data_path = "/storage/research/aiub_gravdet/WD_XOV/"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
in_folder1 = f"{data_path}pyXover/out/{id1}/"
# in_folder1 = f"{in_folder1}3res_20amp/"
out_folder = f"{data_path}OBS/"
# list_arcs = [ file.split('_')[1:] for file in os.listdir(f'{out_folder}xov/*pkl')]
list_file = os.listdir(f'{in_folder1}xov/')
list_arcs = [file.split(".")[-2].split('_')[1:] for file in list_file if file.split(".")[-1]=='pkl']
pyout_folder = f"{data_path}pyXover/out/"
if id1 == id2:
   ids = [id1]
else:
   ids = [id1,id2]

if False:
   startInit = time.time()
   track_list1 = getgtrackt0(pyout_folder, id1)
   track_list2 = getgtrackt0(pyout_folder, id2)
   endInit = time.time()
   print(endInit-startInit)

if False:
   arc1 = 310501
   arc2 = 310501
   csv_filename = f"{out_folder}xov_{id1}{arc1}_{arc2}.csv"

startInit = time.time()


for arc1,arc2 in list_arcs:
   csv_filename = f"{out_folder}xov_{id1}{arc1}_{arc2}.csv"
   # xov_pkl2csv(pyout_folder, csv_filename, arc1, arc2, ids,[track_list1,track_list2])
   # xov_pkl2csv2(pyout_folder, csv_filename, arc1, arc2, ids)
   xov_pkl2csv2(pyout_folder, csv_filename, arc1, arc2, ids, xov_time = False,  distance = False, derivatives = False)
endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")