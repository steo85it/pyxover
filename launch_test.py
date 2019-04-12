import PyAltSim
import PyXover
import AccumXov

import itertools as itert
import os
import shutil

if __name__ == '__main__':

    exp = 'mlatimes'

    res = [0,1,2,3,4,5,6]
    ampl = [1,5,10,25,50]
    # res = [0,1,2,3,4,5,6]
    # ampl = [10, 25, 50]

    cmb = list(
        itert.product(ampl,res))


    dirnams = ['../data/SIM_1301/'+exp+'/'+str(x[1])+'res_'+str(x[0])+'amp_tst/' for x in cmb]
    args_pyaltsim = [(i, j, k) for ((i, j), k) in zip(cmb, dirnams)]


    indirnams = ['SIM_1301/'+exp+'/'+str(x[1])+'res_'+str(x[0])+'amp_tst/' for x in cmb]
    outdirnams = ['sim_'+exp+'/'+str(x[1])+'res_'+str(x[0])+'amp/' for x in cmb]
    args_pyxover = [(9, k, l) for (k, l) in zip(indirnams, outdirnams)]

    # for ie in range(len(args_pyaltsim)):
    #     print("Running PyAltSim with ",args_pyaltsim[ie],"...")
    #     PyAltSim.main(args_pyaltsim[ie])
    #     print("Running PyXover with ",args_pyxover[ie],"...")
    #     PyXover.main(args_pyxover[ie])

    # [PyAltSim.main(x) for x in args_pyaltsim]
    #
    # [PyXover.main(x) for x in args_pyxover]


    AccumXov.main(outdirnams)

    os.rename("tmp/tst.png", "tmp/rms_vs_exp_"+exp+".png")
