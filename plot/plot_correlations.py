
#!/usr/bin/env python3
import glob
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from accumxov.accum_utils import compute_correlations
from scipy.sparse import diags, vstack
import scipy.linalg as la


def plot_correlation_matrix(R, labels=None, title="Parameter Correlation Matrix", 
                            figsize=(6,5), cmap="coolwarm", vmin=-1, vmax=1, fig_name=""):
    """
    Plot a correlation matrix as a heatmap with colorbar and annotations.

    Parameters
    ----------
    R : ndarray (n x n)
        Correlation matrix (values between -1 and +1).
    labels : list of str, optional
        Parameter names for x and y axes.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size (default = (6, 5)).
    cmap : str, optional
        Colormap (default = "coolwarm").
    vmin, vmax : float, optional
        Limits for the color scale (default = -1, 1).
    """
    n = R.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(R, cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient", rotation=270, labelpad=15)

    # Set labels
    if labels is not None:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add gridlines
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate with correlation values
    for i in range(n):
        for j in range(n):
            value = R[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_name)


id = "CC6"
iter = 3
suffix = "_A"
data_path = "/storage/research/aiub_gravdet/WD_BELA/"

startInit = time.time()

pyout_folder = f"{data_path}pyXover/out/{id}_{iter}"
with open(f"{pyout_folder}/Abmat_{id}_{iter}_{iter+1}{suffix}.pkl", "rb") as file:
   Abmat = pickle.load(file)

sigma0 = Abmat.resid_wrmse
spA = sum([np.sqrt(w) * diags(mask_obs.astype(float)) @ Abmat.spA_sol4 for (w,mask_obs) in zip(Abmat.vce_obs,Abmat.obs_blocks)])
spQ = vstack([np.sqrt(w) * diags(p.diagonal()**0.5) for w, p in zip(Abmat.vce_pen, Abmat.penalty_mat)])
# spQ = vstack([np.sqrt(w) * p for w, p in zip(Abmat.vce_pen, Abmat.penalty_mat)])

if len(Abmat.penalty_mat) > 0:
   spA = vstack([spA, 1. * spQ])

N  = spA.T * spA

N = N.todense()
if N.shape[0] < 25000:
   L = la.cholesky(N, lower=True, overwrite_a=True)
N = None

R, Cx= compute_correlations(N=N, L=L, sigma0=sigma0)

N = R.shape[0]
print(N)

ng = 5

ACR_lab = ['A','C','R']
for (i,name) in enumerate(Abmat.sol4_pars[N-ng:]):
   # Rtrack = np.abs(R[N-ng+i,:N-ng])
   Rtrack = R[N-ng+i,:N-ng]
   fig, axs = plt.subplots(3,figsize=(6.1,4.8))
   for j in range(0,3):
      axs[j].hist(Rtrack[j::3],50)
      axs[j].set_xlim([-0.06, 0.15])
      axs[j].set_ylabel(ACR_lab[j])
   axs[2].set_xlabel('Correlation')
   fig_name=f"trcorrel{name[3:]}"
   fig.suptitle(name[4:])
   plt.savefig(f"{fig_name}.png")
   
fig_name = f"correlation_{id}_{iter}{suffix}.png"

plot_correlation_matrix(R[N-ng:,N-ng:], fig_name=fig_name, labels=[l[4:] for l in Abmat.sol4_pars[N-ng:]])



endInit = time.time()
print(f"Finished after {str(endInit-startInit)}s")