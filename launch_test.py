import itertools as itert
import os
import sys

import AccumXov
# own libs
import PyAltSim
import PyXover
import PyGeoloc

if __name__ == '__main__':

    #    arg_names = ['command', 'subset', 'y', 'operation', 'option']
    #    args = dict(zip(arg_names, sys.argv))

    #    You could even use it to generate a namedtuple with values that default to None -- all in four lines!

    #    Arg_list = collections.namedtuple('Arg_list', arg_names)
    #    args = Arg_list(*(args.get(arg, None) for arg in arg_names))

    local = 0
    data_sim = 'data' #'sim'  # 
    exp = '' # 'mlatimes/1301' #  '1s' #'mladata' #
    ext_iter = 0  # external iteration
    # exp += '_'+str(ext_iter)

    # res = [0, 1, 2, 3, 4, 5, 6]
    # ampl = [1, 5, 10, 25, 50]
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

    print("Input args: topo res/ampl ",resampl,", sub-dataset",subarg,", geoloc/xovers/accum", sect)

    if data_sim == 'data':
        res = [0]
        ampl = [1]

    cmb = list(
        itert.product(ampl, res))

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

        # add option to spread over the cluster
        idx_tst = [i for i in range(len(cmb))]
        if local == 0:
            ie = int(resampl)
            if data_sim == 'sim' and ext_iter == 0:
              print("Running PyAltSim with ", args_pyaltsim[ie], "...")
              # PyAltSim.main(args_pyaltsim[ie])
            print("Running PyGeoloc with ", args_pygeoloc[ie], "...")
            PyGeoloc.main(args_pygeoloc[ie])

        else:
            for ie in range(len(args_pyxover)):
                if data_sim == 'sim' and ext_iter == 0:
                    print("Running PyAltSim with ", args_pyaltsim[ie], "...")
                    # PyAltSim.main(args_pyaltsim[ie])
                print("Running PyGeoloc with ", args_pygeoloc[ie], "...")
                PyGeoloc.main(args_pygeoloc[ie])

        # alternative ordering of operations
        # [PyAltSim.main(x) for x in args_pyaltsim]
        # [PyGeoloc.main(x) for x in args_pygeoloc]
        # [PyXover.main(x) for x in args_pyxover]

    elif sect == 2:

        ie = int(resampl)
        # after all tracks have been simulated and geolocalised (if it applies)
        print("Running PyXover with ", args_pyxover[ie], "...")
        PyXover.main(args_pyxover[ie])

    elif sect == 3:

        #outdirnams = [i+'xov/' for i in outdirnams][0]
        AccumXov.main([outdirnams, data_sim, ext_iter])
        if os.path.isfile("tmp/tst.png"):
            os.rename("tmp/tst.png", "tmp/rms_vs_exp_" + exp.split('/')[0] + ".png")

    else:
    
        print("Wrong sect option ", sect)
        exit(2)
