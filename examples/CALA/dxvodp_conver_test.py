
#!/usr/bin/env python3
# # from xovutils.astrotrans import rsw_2_xyz # en vectoriel
# Convert covers partials to ascii file to be used in th Bernese GNSS Software
import os
from  wd_utils import xov_pkl2csv, getgtrackt0, xov_pkl2csv2
import time

id1 = "CA4"
id2 = "CA4"
# data_path = "/storage/homefs/desprats/pyxover/examples/BELA/data/"
# data_path = "/storage/research/aiub_gravdet/WD_XOV/"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"
in_folder1 = f"{data_path}pyXover/out/{id1}_0/"
# in_folder1 = f"{in_folder1}3res_20amp/"
out_folder = f"{data_path}OBS/"
# list_arcs = [ file.split('_')[1:] for file in os.listdir(f'{out_folder}xov/*pkl')]
list_file = os.listdir(f'{in_folder1}xov/')
list_arcs = [file.split(".")[-2].split('_')[1:] for file in list_file if file.split(".")[-1]=='pkl']
pyout_folder = f"{data_path}pyXover/out/"
arc1 = 310501
arc2 = 310501
ids = []
ids.append(id1)
ids = [id1,id2]
# ids = ["CO3"]
if False:
   startInit = time.time()
   track_list1 = getgtrackt0(pyout_folder, id1)
   track_list2 = getgtrackt0(pyout_folder, id2)
   endInit = time.time()
   print(endInit-startInit)
startInit = time.time()
csv_filename = f"{out_folder}xov_{id1}{arc1}_{arc2}.csv"
#arc1 = 2603
#arc2 = 2603
# csv_filename = f"{out_folder}xov_{id1}20-5{arc1}_{arc2}.csv"
# xov_pkl2csv2(pyout_folder, csv_filename, arc1, arc2, [id1])
# xov_pkl2csv(pyout_folder, csv_filename, arc1, arc2, ids,[track_list1,track_list2])
# xov_pkl2csv(pyout_folder, csv_filename, arc1, arc2, ids,[track_list1])
#list_arcs = ['2603', '2603']
for arc1,arc2 in list_arcs:
   csv_filename = f"{out_folder}xov_{id1}{arc1}_{arc2}.csv"
   # xov_pkl2csv(pyout_folder, csv_filename, arc1, arc2, ids,[track_list1,track_list2])
   xov_pkl2csv2(pyout_folder, csv_filename, arc1, arc2, ids)
endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")