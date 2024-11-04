import os
import subprocess
from p_tqdm import p_umap
import pandas as pd


# Scope:
# Download the spice kernels needed
# NOTE: At the moment this script works if either wget or curl are installed in the system

def download_kernels():

    # From PDS
    todownload = ['spk/de421.bsp',
                  'fk/moon_assoc_pa.tf',
                  'fk/moon_080317.tf',
                  'pck/moon_pa_de421_1900_2050.bpc',
                  'pck/pck00010.tpc',
                  'lsk/naif0012.tls',
                  ]

    for f in todownload:
        fname = os.path.basename(f)
        if not os.path.exists(f'examples/kernels/{fname}'):
            try:
                subprocess.run(
                    f'wget -P examples/kernels/  https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f}',
                    shell=True)
            except:
                subprocess.run(
                    f'curl https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f} -o examples/kernels/{fname}',
                    shell=True)

def download_mlasci(csvindex):

    print(csvindex)
    df = pd.read_csv(csvindex, sep=':', header=None)
    df = df.loc[:, 5]
    df = df.loc[df.str.contains('.tab')].reset_index(drop=True)
    df = df.loc[df.str.contains('mlascirdr1201')]
    print(df)

    def download_kernels(f):

        fname = os.path.basename(f)
        if not os.path.exists(f'examples/MLA/data/raw/SIM_12/BS0/0res_1amp/{fname}'):
            try:
                subprocess.run(
                    f'wget -P examples/MLA/data/raw/SIM_12/BS0/0res_1amp/  https://pds-geosciences.wustl.edu/messenger/mess-e_v_h-mla-3_4-cdr_rdr-data-v1/messmla_2001/rdr_radr/2012/jan/{f}',
                    shell=True)
            except:
                subprocess.run(
                    f'curl https://pds-geosciences.wustl.edu/messenger/mess-e_v_h-mla-3_4-cdr_rdr-data-v1/messmla_2001/rdr_radr/2011/jan/{f} -o examples/MLA/data/raw/SIM_11/BS0/0res_1amp/{fname}',
                    shell=True)

    p_umap(download_kernels, df.values)

if __name__ == '__main__':

    csvindex = "/home/sberton2/Scaricati/collection_data_rdr_radr_inventory.csv"
    download_mlasci(csvindex)