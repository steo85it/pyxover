import itertools as itert
import os
import sys

import AccumXov
import PyAltSim
# own libs
import PyXover

if __name__ == '__main__':

#    arg_names = ['command', 'subset', 'y', 'operation', 'option']
#    args = dict(zip(arg_names, sys.argv))

#    You could even use it to generate a namedtuple with values that default to None -- all in four lines!

#    Arg_list = collections.namedtuple('Arg_list', arg_names)
#    args = Arg_list(*(args.get(arg, None) for arg in arg_names))

    local = 0
    data_sim = 'sim'  # 'data' #  !! change dataset in AccumXov.load/combine !!
    exp = 'mlatimes'  # '1s' #'mladata' # !! change PyAltSim IllumNG source file

    if len(sys.argv) > 2:
      epos = sys.argv[2]
    else:
      epos = '1201'

    if len(sys.argv) > 3:
      subxov = sys.argv[3]
    else:
      subxov = '9'

    # res = [0, 1, 2, 3, 4, 5, 6]
    # ampl = [1, 5, 10, 25, 50]
    res = [0]
    ampl = [1]

    if data_sim == 'data':
        res = [0]
        ampl = [1]

    cmb = list(
        itert.product(ampl, res))

    if data_sim == 'sim':
        if local:
            dirnams = ['../data/SIM_' + epos + '/' + exp + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp_tst/' for x in
                       cmb]
        else:
            dirnams = ['/att/nobackup/sberton2/MLA/data/SIM_' + epos[:2] + '/' + exp + '/' + str(x[1]) + 'res_' + str(
                x[0]) + 'amp_tst/' for x in cmb]

        args_pyaltsim = [(i, j, k, epos) for ((i, j), k) in zip(cmb, dirnams)]

        indirnams = ['SIM_'+epos[:2]+'/' + exp + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp_tst/' for x in cmb]
        outdirnams = ['sim_' + exp + '/' + str(x[1]) + 'res_' + str(x[0]) + 'amp/' for x in cmb]
        args_pyxover = [(subxov, k, l, 'MLASIMRDR') for (k, l) in zip(indirnams, outdirnams)]
    else:
        indirnams = ['1301/']
        outdirnams = ['real/1301/']
        args_pyxover = [(subxov, k, l, 'MLASCIRDR') for (k, l) in zip(indirnams, outdirnams)]

    # add option to spread over the cluster
    idx_tst = [i for i in range(len(cmb))]
    # if len(sys.argv) > 1 and local == 0:
    #
    #     ie = int(sys.argv[1])
    #     if data_sim == 'sim':
    #         print("Running PyAltSim with ", args_pyaltsim[ie], "...")
    #         PyAltSim.main(args_pyaltsim[ie])
    #     print("Running PyXover with ", args_pyxover[ie], "...")
    #     PyXover.main(args_pyxover[ie])
    #     exit()
    #
    # else:
    #     for ie in range(len(args_pyxover)):
    #         if data_sim == 'sim':
    #             print("Running PyAltSim with ", args_pyaltsim[ie], "...")
    #             PyAltSim.main(args_pyaltsim[ie])
    #         print("Running PyXover with ", args_pyxover[ie], "...")
    #         PyXover.main(args_pyxover[ie])

    # alternative ordering of operations
    [PyAltSim.main(x) for x in args_pyaltsim]
    # [PyXover.main(x) for x in args_pyxover]
    exit()

    AccumXov.main([outdirnams, data_sim])
    os.rename("tmp/tst.png", "tmp/rms_vs_exp_" + exp + ".png")
