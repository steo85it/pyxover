from src.accumxov.Amat import Amat
from examples.MLA.options import outdir, vecopts, tmpdir
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plt_histo(series, xlim='', filename='test'):

    if xlim == '':
        xlim = series.abs().max()

    if filename == '0':
        plt.figure(figsize=(8,3))
    plt.xlim(-1.*xlim, xlim)
    # the histogram of the data
    num_bins = 200 # 'auto'
    n, bins, patches = plt.hist(series.astype(np.float), bins=num_bins, density=True, facecolor='blue',
    alpha=1./(float(filename)+1.), range=[-1.*xlim, xlim],cumulative=False)

    plt.xlabel('dR (m)')
    plt.ylabel('Probability')
    plt.title('Histogram of '+ filename) #dR: $\mu=' + str(mean_dR) + ', \sigma=' + str(std_dR) + '$')
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(tmpdir+'/weights_histo_' + filename + '.png')
    # plt.clf()

def get_weights(iter):

    sol_list = [outdir + 'sim/archived/KX1r4_KX/KX1r4_'+str(iter)+'/0res_1amp/Abmat_sim_KX1r4_'+str(iter+1)+'_0res_1amp.pkl',
        #[outdir + 'Abmat/KX1r4_AG/KX1r4_'+str(iter)+'/0res_1amp/Abmat_sim_KX1r4_'+str(iter+1)+'_0res_1amp.pkl',
                # outdir + 'Abmat/KX1r4_IAU2/KX1r4_'+str(iter)+'/0res_1amp/Abmat_sim_KX1r4_'+str(iter+1)+'_0res_1amp.pkl']
                outdir + 'sim/KX1_'+str(iter)+'/0res_1amp/Abmat_sim_KX1_' + str(iter + 1) + '_0res_1amp.pkl']

    # np.sort(glob.glob(outdir+'sim/'+exp+'_'+str(iter)+'/3res_20amp/Abmat_sim_'+exp+'_'+str(iter+1)+'_3res_20amp.pkl'))
    list_exp = []

    for idx,sol in enumerate(sol_list):
        # print("Processing", sol)
        prev = Amat(vecopts)
        solmat = prev.load(sol)

        tmp = solmat.xov.xovers[['orbA', 'orbB', 'weights']].copy()
        print("nb of xovers:", idx, len(tmp))
        tmp['orbs'] = tmp['orbA'].map(str) + '-' + tmp['orbB']
        tmp.drop(['orbA','orbB'],axis=1,inplace=True)

        list_exp.append(tmp.set_index('orbs'))

    compare_df = pd.concat(list_exp,axis=1,sort=False).reset_index()
    compare_df.columns = ['orbs','weights_A','weights_B']
    print("nr of null: A=", compare_df.weights_A.isnull().sum(),", B=", compare_df.weights_B.isnull().sum())
    compare_df['wdiff'] = (compare_df.weights_A - compare_df.weights_B)/compare_df.weights_A*100.
    # print(compare_df.loc[compare_df['wdiff']==None])
    compare_df.dropna(inplace=True)
    # compare_df.fillna(1000000,inplace=True)

    plt_histo(series=compare_df['wdiff'],xlim=100.,filename=str(iter))



if __name__ == '__main__':

    pd.set_option('mode.use_inf_as_na', True)

    itermax = 9

    for iter in range(itermax):
        get_weights(iter)
