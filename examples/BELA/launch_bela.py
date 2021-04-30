from accumxov import AccumXov
from accumxov.Amat import Amat
from pygeoloc import PyGeoloc
from pyxover import PyXover
from config import XovOpt

# PyTest requires parallel = False
if __name__ == '__main__':

        # update paths and check options
        XovOpt.set("basedir", 'data/')
        XovOpt.set("instrument", 'BELA')

        XovOpt.set("sol4_orb", [])
        XovOpt.set("sol4_orbpar", [None])
        XovOpt.set("sol4_glo", ['dR/dRA', 'dR/dDEC', 'dR/dPM', 'dR/dL','dR/dh2'])

        vecopts = {'SCID': '-121',  # '-236',
                   'SCNAME': 'MPO',  # 'MESSENGER',
                   'SCFRAME': -121000,  # -236000,
                   # 'INSTID': (-236500, -236501),
                   # 'INSTNAME': ('MSGR_MLA', 'MSGR_MLA_RECEIVER'),
                   'PLANETID': '199',
                   'PLANETNAME': 'MERCURY',
                   'PLANETRADIUS': 2440.,
                   'PLANETFRAME': 'IAU_MERCURY',
                   'OUTPUTTYPE': 1,
                   'ALTIM_BORESIGHT': '',
                   'INERTIALFRAME': 'J2000',
                   'INERTIALCENTER': 'SSB',
                   'PM_ORIGIN': 'J2013.0',
                   'PARTDER': ''}
        XovOpt.set("vecopts", vecopts)

        XovOpt.set("expopt", 'BE0')
        XovOpt.set("resopt" , 3)
        XovOpt.set("amplopt" , 20)
        XovOpt.set("SpInterp", 0)
        XovOpt.set("spauxdir", 'MPO_spk/')
        XovOpt.set("parallel", True)
        XovOpt.check_consistency()

        # run full pipeline on a few MLA test data
        months_to_process = ['2604','2612']

        # for monyea in months_to_process:
        #     indir_in = f'SIM_{monyea[:2]}/{XovOpt.get("expopt")}/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'
        #     outdir_in = f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_{monyea[:2]}'
        #     # geolocation step
        #     PyGeoloc.main([f'{monyea}', indir_in, outdir_in, 'MLASCIRDR', 0])
        # # crossovers location step
        # PyXover.main(['0', f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/gtrack_',
        #               f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/', 'MLASIMRDR', 0])
        # lsqr solution step
        out = AccumXov.main([[f'sim/{XovOpt.get("expopt")}_0/{XovOpt.get("resopt")}res_{XovOpt.get("amplopt")}amp/'], 'sim', 0])

        # # generate new template (when needed)
        # # out.save('mla_pipel_test_out.pkl')
        #
        # # load template test results
        # val = Amat(vecopts=XovOpt.get("vecopts"))
        # try:
        #     val = val.load('mla_pipel_test_out.pkl')
        # except:
        #     val = val.load('tests/MLA/mla_pipel_test_out.pkl')
        #
        # # check xovers residuals
        # print(out)
        # # round up to avoid issues with package updates
        # res_out = [round(x, 4) for x in out.b]
        # res_val = [round(x, 4) for x in val.b]
        #
        # # perform test
        # self.assertEqual(res_out, res_val)
        #
        # # check parameter solutions
        # # round up to avoid issues with package updates
        # out = {key : round(out.sol_dict['sol'][key], 4) for key in out.sol_dict['sol']}
        # val = {key : round(val.sol_dict['sol'][key], 4) for key in val.sol_dict['sol']}
        #
        # # perform test
        # self.assertEqual(out, val)
