import itertools as itert
import os
import shutil
import sys

from prOpt import sim_altdata, local, outdir

# Experiments
#------------
# tp2: full dataset, simulation, no perturbations (0-test)
# tp3: full dataset, simulation, h2 = 0, no perturbations
# tp4: full dataset, simulation, h2 = 1.0, no perturbations, noise, PM@J2013.0
# tp8: full dataset, simulation, h2 = 0.8, no perturbations (as tp2, but updated code --> eg, 0.5 in tid)
# tp9: full dataset, simulation, h2 = 0.8, no perturbations, noise, PM@J2013.0
# KX1: real data, tidal h2=0, KX orbits+IAU ap, 1mln xovers (h2 since the beginning)
# KX2: real data, KX orbits+IAU ap, 1mln xovers (h2 since convergence)
# KX3: real data, KX orbits+AG ap, 1mln xovers (h2 since convergence) - wrong AbMat name, just rename
# KX1r2: real data, same subsel as tp8 (apriori h2=0 after 05-Feb)
# tpAp: full simu, KX orbits, AG a priori

#@profile
def main():

    data_sim = 'sim'  # 'data' #
    exp = 'AGTP' # 'tp4' #
    # exp += '_'+str(ext_iter)

#    res = [3]
#    ampl = [5,10,20,30,40,60,80]
    res = [0]
    ampl = [1]

    if len(sys.argv) > 1:
        resampl = sys.argv[1]
    else:
        resampl = '0'

    if len(sys.argv) > 2:
        subarg = sys.argv[2]
    else:
        subarg = '1201' # or 9 if pyxover

    if len(sys.argv) > 3:
        sect = int(sys.argv[3])
    else:
        sect = -1

    if len(sys.argv) > 4:
        ext_iter = int(sys.argv[4])
    else:
        ext_iter = 0  # external iteration

    print("Input args: topo res/ampl ",resampl,", sub-dataset",subarg,
          ", geoloc/xovers/accum", sect, ", ext_iter", ext_iter)

    if data_sim == 'data':
        res = [0]
        ampl = [1]

    cmb = list(
        itert.product(ampl, res))
    # print(dict(zip(range(len(cmb)),cmb)))

    if data_sim == 'sim':
        if local:
            dirnams = ['../data/SIM_' + subarg[:2] + '/' + exp + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp/' for x in
                       cmb]
        else:
            dirnams = ['/att/nobackup/sberton2/MLA/data/SIM_' + subarg[:2] + '/' + exp + '/' + str(x[1]) + 'res_' + str(
                x[0]) + 'amp/' for x in cmb]

        args_pyaltsim = [(i, j, k, subarg) for ((i, j), k) in zip(cmb, dirnams)]

        indirnams = ['SIM_' + subarg[:2] + '/' + exp + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp/' for x in cmb]
        outdirnams = ['sim/' + exp + '_' + str(ext_iter) + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp/gtrack_'+ subarg[:2] for x in cmb]
        args_pygeoloc = [(subarg, k, l, 'MLASIMRDR',ext_iter) for (k, l) in zip(indirnams, outdirnams)]

        indirnams = ['sim/' + exp + '_' + str(ext_iter) + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp/gtrack_' for x in cmb]
        outdirnams = ['sim/' + exp + '_' + str(ext_iter) + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp/' for x in cmb]
        args_pyxover = [(subarg, k, l, 'MLASIMRDR',ext_iter) for (k, l) in zip(indirnams, outdirnams)]

    else:
    
        indirnams = ['MLA_' + subarg[:2] + '/']
        outdirnams = ['mladata/' + exp + 'gtrack_'+ subarg[:2]]
        args_pygeoloc = [(subarg, k, l, 'MLASCIRDR',ext_iter) for (k, l) in zip(indirnams, outdirnams)]
    
        indirnams = ['mladata/' + exp + 'gtrack_']
        outdirnams = ['mladata/'+exp]
        args_pyxover = [(subarg, k, l, 'MLASCIRDR',ext_iter) for (k, l) in zip(indirnams, outdirnams)]

    if sect == 1:
        import PyAltSim
        import PyGeoloc

        # save options file to outdir
        if not os.path.exists(outdir+outdirnams[0]):
            os.makedirs(outdir+outdirnams[0], exist_ok=True)
        shutil.copy(os.getcwd()+'/prOpt.py', outdir+outdirnams[0])

        # add option to spread over the cluster
        idx_tst = [i for i in range(len(cmb))]
        if local == 0:
            ie = int(resampl)
            if data_sim == 'sim' and sim_altdata and ext_iter == 0:
              print("Running PyAltSim with ", args_pyaltsim[ie], "...")
              PyAltSim.main(args_pyaltsim[ie])
              exit(0)
            print("Running PyGeoloc with ", args_pygeoloc[ie], "...")
            PyGeoloc.main(args_pygeoloc[ie])

        else:
            for ie in range(len(args_pyaltsim)):
                if data_sim == 'sim' and sim_altdata and ext_iter == 0:
                    ie = int(resampl)
                    print("Running PyAltSim with ", args_pyaltsim[ie], "...")
                    PyAltSim.main(args_pyaltsim[ie])
                    exit(0)
                print("Running PyGeoloc with ", args_pygeoloc[ie], "...")
                PyGeoloc.main(args_pygeoloc[ie])

        # alternative ordering of operations
        # [PyAltSim.main(x) for x in args_pyaltsim]
        # [PyGeoloc.main(x) for x in args_pygeoloc]
        # [PyXover.main(x) for x in args_pyxover]

    elif sect == 2:
        import PyXover

        ie = int(resampl)
        # after all tracks have been simulated and geolocalised (if it applies)
        print("Running PyXover with ", args_pyxover[ie], "...")
        PyXover.main(args_pyxover[ie])

    elif sect == 3:
        import AccumXov

        #outdirnams = [i+'xov/' for i in outdirnams][0]
        AccumXov.main([outdirnams, data_sim, ext_iter])
        if os.path.isfile("tmp/tst.png"):
            os.rename("tmp/tst.png", "tmp/rms_vs_exp_" + exp.split('/')[0] + ".png")

    else:
    
        print("Wrong sect option ", sect)
        print("How to call: launch_test.py sim_terrain epoch proc_step iter")
        print("where:\n sim_terrain indicates roughness terrain for simulations (elements in res/amp list)\n"
              "epoch is YYMM if proc_step=1, combination of months and year in list of cmb if proc_step==2, whatever if proc_step==3\n"
              "proc_step can be 1=geolocalise data, 2=locate and process xovers, 3=cumulate and invert")
        exit(2)

if __name__ == '__main__':

    main()
