import glob
import os

from Amat import Amat
from ground_track import gtrack
from prOpt import outdir, vecopts, partials, parOrb, parGlo, parallel
import numpy as np
import pandas as pd
import time
from project_coord import project_stereographic
from xov_setup import xov
import multiprocessing as mp
# from memory_profiler import profile

msrm_smpl = 4  # should be even...

#@profile
def fine_xov_proc(xovi,df,xov_tmp): #args):
    # xovi = args[0]
    # df = args[1]
    # xov_tmp = args[2]

    # print(proj_df_xovi.columns)
    # print(len(proj_df_xovi))
    # exit()

    # if (idx / len(xovs_list) * 100.) % 5 == 0:
    #     print("Working... ", (idx / len(xovs_list) * 100.), "% done ...")

    # try:

    df = df.reset_index(
        drop=True).reset_index()
    df.rename(columns={'index': 'genID', 'seqid_mla': 'seqid'}, inplace=True)

    if len(df)>0: # and {'LON_proj','LAT_proj'}.issubset(df.columns):
        xov_tmp.proj_center = {'lon': df['LON_proj'].values[0], 'lat': df['LAT_proj'].values[0]}
        df.drop(columns=['LON_proj','LAT_proj'],inplace=True)
    else:
        print("### Empty proj_df_xovi on xov #", xovi)
        print(df)
        return  # continue

    xov_tmp.ladata_df = df.copy()

    xov_tmp.msrm_sampl = msrm_smpl
    xov_tmp.tracks = dict(zip(df.orbID.unique(), list(range(2))))


    xov_tmp.ladata_df['orbID'] = xov_tmp.ladata_df['orbID'].map(xov_tmp.tracks)

    try:
        x, y, subldA, subldB, ldA, ldB = xov_tmp.get_xover_fine([msrm_smpl], [msrm_smpl], '')
    except:
        print("### get_xover_fine issue on xov #", xovi)
        print(xov_tmp.ladata_df)
        return  # continue
    # print(x, y, subldA, subldB, ldA, ldB)

    ldA, ldB, R_A, R_B = xov_tmp.get_elev('', subldA, subldB, ldA, ldB, x=x, y=y)
    # print(ldA, ldB, R_A, R_B)
    out = np.vstack((x, y, ldA, ldB, R_A, R_B)).T
    # print(df)
    # print(out)
    if len(out) == 0:
        print("### Empty out on xov #", xovi)
        return  # continue

    # post-processing
    xovtmp = xov_tmp.postpro_xov_elev(xov_tmp.ladata_df, out)

    if len(xovtmp) > 1:
        print("### Bad multi-xov at xov#", xovi)
        print(xovtmp)
        return  # continue

    # Update xovtmp as attribute for partials
    xov_tmp.xovtmp = xovtmp

    # Compute and store distances between obs and xov coord
    xov_tmp.set_xov_obs_dist()
    # Compute and store offnadir state for obs around xov
    xov_tmp.set_xov_offnadir()

    # process partials from gtrack to xov, if required
    if partials:
        xov_tmp.set_partials()

    # Remap track names to df
    xov_tmp.xovtmp['orbA'] = xov_tmp.xovtmp['orbA'].map({v: k for k, v in xov_tmp.tracks.items()})
    xov_tmp.xovtmp['orbB'] = xov_tmp.xovtmp['orbB'].map({v: k for k, v in xov_tmp.tracks.items()})

    xov_tmp.xovtmp['xOvID'] = xovi

    # print("inside", xov_tmp.xovtmp)

    # Update general df (serial only, does not work in parallel since not a shared object)
    if not parallel:
        xov_tmp.xovers = xov_tmp.xovers.append(xov_tmp.xovtmp, sort=True)

    # import sys
    # def sizeof_fmt(num, suffix='B'):
    #     ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    #     for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
    #         if abs(num) < 1024.0:
    #             return "%3.1f %s%s" % (num, unit, suffix)
    #         num /= 1024.0
    #     return "%.1f %s%s" % (num, 'Yi', suffix)
    #
    # print("iter over xov:",xovi)
    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
    #                          key=lambda x: -x[1])[:10]:
    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    # exit()

    return xov_tmp.xovtmp


def get_xov_latlon(xov, df):
    """
    Retrieve LAT/LON of xover from geolocalisation table and add to xovers table
    (useful to analyze/plot xovers on map). Updates input xov.xovers.

    :param xov: contains xovers table to merge to
    :param df: mla data table with lat and lon to extract
    """
    # pd.set_option('display.max_columns', 500)

    # only get rows corresponding to former xov location
    df = df.loc[df.seqid_xov == df.seqid_mla][['xovid', 'LON', 'LAT']]
    # will get 2 rows, one for each track in the xover: drop one
    df = df.drop_duplicates(subset=['xovid'], keep='first')
    # assign LAT and LON to their respective xover by id and update
    xov.xovers = pd.merge(xov.xovers, df, left_on='xOvID', right_on='xovid')
    # xov.xovers.drop('xovid',inplace=True)

    return xov

def project_chunk(proc_chunk):

    chunk_res = project_stereographic(proc_chunk['LON'],
                                      proc_chunk['LAT'],
                                      proc_chunk['LON_proj'],
                                      proc_chunk['LAT_proj'],
                                      R=vecopts['PLANETRADIUS'])

    proc_chunk[['x','y']] = pd.DataFrame(chunk_res).T
    # chunk_res.index = proc_chunk.index

    return proc_chunk

#@profile
def xov_prc_iters_run(args,cmb):
    start = time.time()

    xov_iter = args[-1]
    outdir_in = args[2] #"sim/KX1r4_0/0res_1amp/"

    if msrm_smpl % 2 != 0:
        print("*** ERROR: msrm_smpl not an even number:", msrm_smpl)

    outdir_old = outdir_in.replace('_' + str(xov_iter) + '/', '_' + str(xov_iter - 1) + '/')
    # print(outdir_old, outdir_in)
    tmp = Amat(vecopts)
    # print(outdir + outdir_old + 'Abmat*.pkl')
    tmp = tmp.load(glob.glob(outdir + outdir_old + 'Abmat*.pkl')[0])

    # print(tmp.xov.xovers.columns)
    old_xovs = tmp.xov.xovers[['LON', 'LAT', 'xOvID', 'orbA', 'orbB', 'mla_idA', 'mla_idB']]
    old_xovs = old_xovs.loc[
        (old_xovs['orbA'].str.startswith(str(cmb[0]))) & (old_xovs['orbB'].str.startswith(str(cmb[1])))]
    # print(tmp.xov.xovers.loc[tmp.xov.xovers.xOvID.isin([7249,7526,1212,8678,34,11436,625])][['R_A','R_B','x0','y0']])
    # exit()

    tracks_in_xovs = np.unique(old_xovs[['orbA', 'orbB']].values)
    print("Processing",len(tracks_in_xovs),"tracks, previously resulting in", len(old_xovs),"xovers.")

    track = gtrack(vecopts)

    proj_input = []
    mla_idx = ['mla_idA', 'mla_idB']
    for track_id in tracks_in_xovs[:]:
        # if track_id in ['1312171723','1312240123']:
        # print(track_id)

        for idx, orb in enumerate(['orbA', 'orbB']):
            # print(idx,orb)
            # print(outdir + outdir_in + 'gtrack_' + str(cmb[idx]) + '/gtrack_' + track_id + '.pkl')

            # load file if year of track corresponds to folder
            if track_id[:2]==str(cmb[idx]):
                track = track.load(outdir + outdir_in + 'gtrack_' + str(cmb[idx]) + '/gtrack_' + track_id + '.pkl')
            else:
                # print("### Track ",track_id,"does not exist in",outdir + outdir_in + 'gtrack_' + str(cmb[idx]))
                # track = gtrack(vecopts) # reinitialize track to avoid error "'NoneType' object has no attribute 'load'"
                continue

            # exit()
            tmp_ladata = track.ladata_df.copy()  # .reset_index()

            # xov df extract and expand to neighb obs
            tmp = old_xovs.loc[old_xovs[orb] == track_id][[mla_idx[idx], 'xOvID', 'LON', 'LAT']].astype(
                {mla_idx[idx]: int, 'xOvID': int, 'LON': float, 'LAT': float}).round({'LON': 3, 'LAT': 3}).values
            # print(tmp)
            # sampl = 10
            xov_extract = pd.DataFrame(tmp, columns=['seqid', 'xovid', 'LON_proj', 'LAT_proj'])
            xov_extract.sort_values(by='seqid', inplace=True)
            # print(xov_extract)
            ########### TEST
            # xov_extract = xov_extract.loc[xov_extract.xovid == 0]
            ###########
            # exit()
            # # get geoloc obs with same seqid
            rows = tmp_ladata['seqid'].loc[tmp_ladata['seqid'].isin(xov_extract.loc[:, 'seqid'])]  # .index)  #
            # check if multiple xovers at same mla index
            multiple_counts = xov_extract.loc[:, 'seqid'].value_counts()  # .loc[lambda x: x>1]

            if len(multiple_counts.loc[lambda x: x > 1]) > 0:
                # print(rows.to_frame())
                tmp = pd.merge(rows.reset_index(), multiple_counts.reset_index(), left_on='seqid',
                               right_on='index').drop(['seqid_x', 'index_y'], axis=1)
                tmp.columns = ["genid", "dupl"]
                rows = tmp.reindex(tmp.index.repeat(tmp.dupl)).set_index('genid')

            xov_extract['index'] = rows.index

            xov_extract = xov_extract[['index', 'seqid', 'xovid', 'LON_proj', 'LAT_proj']]
            # print("xov",xov_extract)

            tmp = xov_extract.values
            mla_close_to_xov = np.ravel([np.arange(i[0] - msrm_smpl, i[0] + (msrm_smpl + 1), 1) for i in tmp])
            tmp = np.hstack(
                [mla_close_to_xov[:, np.newaxis], np.reshape(np.tile(tmp[:, 1:], (msrm_smpl * 2 + 1)), (-1, 4))])
            xov_extract = pd.DataFrame(tmp, columns=['genid', 'seqid', 'xovid', 'LON_proj', 'LAT_proj'])

            ladata_extract = tmp_ladata.loc[tmp_ladata.reset_index()['index'].isin(xov_extract.genid)][
                ['seqid', 'LON', 'LAT', 'orbID', 'ET_BC', 'ET_TX', 'R', 'offnadir']].reset_index()
            # print("ladata",ladata_extract)
            # if idx>0:
            #     exit()
            # print(xov_extract)
            # exit()
            tmp_proj = pd.merge(xov_extract, ladata_extract, left_on='genid', right_on='index')
            tmp_proj['partid'] = 'none'
            # print("len tmp proj",len(tmp_proj))

            if partials:
                # do stuff
                pars = ['d' + x + '/' + y for x in ['LON', 'LAT', 'R'] for y in
                        list(parOrb.keys()) + list(parGlo.keys())]
                delta_pars = {**parOrb, **parGlo}
                etbcs = ['ET_BC_' + x + y for x in list(parOrb.keys()) + list(parGlo.keys()) for y in ['_p', '_m']]
                tmp_ladata_partials = tmp_ladata.loc[tmp_ladata.reset_index()['index'].isin(xov_extract.genid)][
                    ['seqid', 'LON', 'LAT', 'orbID'] + pars + etbcs]
                # print(tmp_ladata_partials.columns)
                # print(delta_pars)
                # update dict keys to multiply derivatives by deltas for LON and LAT and get increments
                delta_pars_tmp = dict(zip(['dLON/' + x for x in delta_pars.keys()],
                                          [np.linalg.norm(x) for x in list(delta_pars.values())]))
                delta_pars_tmp.update(dict(zip(['dLAT/' + x for x in delta_pars.keys()],
                                               [np.linalg.norm(x) for x in list(delta_pars.values())])))
                part_deltas_df = pd.Series(delta_pars_tmp) * tmp_ladata_partials.loc[:, pars].astype(float)
                # use increment to extrapolate new LON and LAT values
                newlon_p = part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLON")].add(
                    tmp_ladata_partials.LON, axis=0)
                newlat_p = part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLAT")].add(
                    tmp_ladata_partials.LAT, axis=0)
                newlon_m = -1. * (
                    part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLON")].sub(tmp_ladata_partials.LON,
                                                                                             axis=0))
                newlat_m = -1. * (
                    part_deltas_df.loc[:, part_deltas_df.columns.str.startswith("dLAT")].sub(tmp_ladata_partials.LAT,
                                                                                             axis=0))
                # update column names
                newlon_p.columns = ['LON_' + x + '_p' for x in delta_pars.keys()]
                newlat_p.columns = ['LAT_' + x + '_p' for x in delta_pars.keys()]
                newlon_m.columns = ['LON_' + x + '_m' for x in delta_pars.keys()]
                newlat_m.columns = ['LAT_' + x + '_m' for x in delta_pars.keys()]

                tmp_proj_part = tmp_proj.drop(['ET_BC', 'LON', 'LAT'], axis=1)
                for pder in ['_' + x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']]:
                    if pder[-1] == 'p':
                        tmp = pd.concat([tmp_proj.set_index('genid')[
                                             ['seqid_x', 'xovid', 'LON_proj', 'LAT_proj', 'index', 'seqid_y', 'orbID',
                                              'offnadir', 'ET_TX']],
                                         tmp_ladata_partials['ET_BC' + pder], tmp_ladata_partials['dR/' + pder[1:-2]],
                                         newlon_p['LON' + pder], newlat_p['LAT' + pder]], axis=1)
                    else:
                        tmp = pd.concat([tmp_proj.set_index('genid')[
                                             ['seqid_x', 'xovid', 'LON_proj', 'LAT_proj', 'index', 'seqid_y', 'orbID',
                                              'offnadir', 'ET_TX']],
                                         tmp_ladata_partials['ET_BC' + pder], tmp_ladata_partials['dR/' + pder[1:-2]],
                                         newlon_m['LON' + pder],
                                         newlat_m['LAT' + pder]], axis=1)
                    tmp['partid'] = pder[1:]
                    tmp.reset_index(inplace=True)
                    tmp.rename(
                        columns={'level_0': 'genid', 'LON' + pder: "LON", 'dR/' + pder[1:-2]: "R", 'LAT' + pder: "LAT",
                                 'ET_BC' + pder: "ET_BC"}, inplace=True)
                    proj_input.append(tmp)

            proj_input.append(tmp_proj)

    end_prepro = time.time()

    print("Pre-processing finished after ", end_prepro - start, "sec!")

    start_proj = time.time()

    proj_df = pd.concat(proj_input, sort=False)

    proj_df.rename(columns={"seqid_x": "seqid_xov", "seqid_y": "seqid_mla"}, inplace=True)

    if parallel:
        n_proc = mp.cpu_count()-1
        n_chunks = 2*n_proc

        proj_df.reset_index(inplace=True,drop=True)
        # this often can't be devided evenly (handle this in the for-loop below)
        chunksize = len(proj_df) // n_chunks

        # devide into chunks
        proc_chunks = []
        for i_proc in range(n_chunks):
            chunkstart = i_proc * chunksize
            # make sure to include the division remainder for the last process
            chunkend = (i_proc + 1) * chunksize if i_proc < n_chunks - 1 else None

            proc_chunks.append(proj_df.iloc[slice(chunkstart, chunkend)])

        assert sum(map(len, proc_chunks)) == len(proj_df)   # make sure all data is in the chunks

        # distribute work to the worker processes
        with mp.get_context("spawn").Pool(processes=n_proc) as pool:
            # starts the sub-processes without blocking
            # pass the chunk to each worker process
            proc_results = [pool.apply_async(project_chunk, args=(chunk,)) for chunk in proc_chunks]
            pool.close()
            pool.join()
            # blocks until all results are fetched
            result_chunks = [r.get() for r in proc_results]
            # print(result_chunks)

        # concatenate results from worker processes
        proj_df = pd.concat(result_chunks)
    else:
        proj_df['x'], proj_df['y'] = zip(*proj_df.apply(
            lambda x: project_stereographic(x['LON'], x['LAT'], x['LON_proj'], x['LAT_proj'],
                                            R=vecopts['PLANETRADIUS']),
            axis=1))  # (lon, lat, lon0, lat0, R=1)

    print("len proj_df:", len(proj_df))

    # split rows related to main observable
    tmp_proj = proj_df.loc[proj_df['partid'] == 'none'].copy()
    tmp_proj.rename(columns={'x': 'X_stgprj', 'y': 'Y_stgprj'},
                    inplace=True)

    # split and re-concatenate rows related to partial derivatives
    if partials:
        partials_df_list = []
        for pder in ['_' + x + '_' + y for x in delta_pars.keys() for y in ['p', 'm']]:
            partials_proj_df = proj_df.loc[proj_df['partid'] == pder[1:]].copy()
            partials_proj_df.rename(columns={'x': 'X_stgprj' + pder, 'y': 'Y_stgprj' + pder, "ET_BC": 'ET_BC' + pder,
                                             "R": 'dR/' + pder[1:-2]},
                                    inplace=True)
            partials_df_list.append(partials_proj_df[['X_stgprj' + pder, 'Y_stgprj' + pder, 'ET_BC' + pder,
                                                      'dR/' + pder[
                                                              1:-2]]])  # could include genid and xovid from partials to check

        proj_df = pd.concat([tmp_proj] + partials_df_list, axis=1, sort=False)
    else:
        proj_df = tmp_proj

    end_proj = time.time()

    print("Projection finished after ", end_proj - start_proj, "sec!")

    start_finexov = time.time()

    xovs_list = proj_df.xovid.unique()
    xov_tmp = xov(vecopts)

    # store involved tracks as dict
    # xov_tmp.tracks = dict(zip(df.orbID.unique(), list(range(2))))
    # store the imposed perturbation (if closed loop simulation) - get from last track uploaded in prepro step
    xov_tmp.pert_cloop = {'0': track.pert_cloop}
    xov_tmp.pert_cloop_0 = {'0': track.pert_cloop_0}
    # store the solution from the previous iteration
    xov_tmp.sol_prev_iter = {'0': track.sol_prev_iter}

    # select what to keep
    cols_to_keep = proj_df.loc[:,
                   proj_df.columns.str.startswith('ET_BC') + proj_df.columns.str.startswith('X_stgprj') +
                   proj_df.columns.str.startswith('Y_stgprj') + proj_df.columns.str.startswith(
                       'dR/')].columns  # ,'X_stgprj','Y_stgprj'))
    proj_df_tmp = proj_df[['orbID', 'seqid_mla', 'ET_TX', 'LON', 'LAT', 'R', 'offnadir','xovid','LON_proj','LAT_proj'] + list(cols_to_keep)]
    # print(proj_df_tmp.columns)

    dfl = []
    for xovi in xovs_list:
        dfl.append(proj_df_tmp.loc[proj_df_tmp['xovid'] == xovi].copy())

    # args = ((xovi, dfl[idx], xov_tmp) for idx in range(len(xovs_list)))  # range(50)) #

    # print(args)
    # exit()

    import sys
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
    #                          key=lambda x: -x[1])[:10]:
    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    if parallel:
        ncores = mp.cpu_count() - 1  # 8
        # pool = mp.Pool(processes=ncores)  # mp.cpu_count())
        # xov_list = pool.map(fine_xov_proc, args)  # parallel
        with mp.get_context("spawn").Pool(processes=ncores) as pool:

            xov_list = [pool.apply_async(fine_xov_proc, args=(xovs_list[idx], dfl[idx], xov_tmp)) for idx in range(len(xovs_list))]
            pool.close()
            pool.join()

        # blocks until all results are fetched
        xov_list = [r.get() for r in xov_list]
        # launch once in serial mode to get ancillary values
        fine_xov_proc(0, dfl[0], xov_tmp)
        # assign xovers to the new xov_tmp containing ancillary values
        xov_tmp.xovers = pd.concat(xov_list, axis=0)

    else:
        for idx in range(len(xovs_list)):
            fine_xov_proc((xovs_list[idx], dfl[idx], xov_tmp))
            # print(idx)
            # xov_tmp.xovers.info(memory_usage='deep')

        # _ = [fine_xov_proc(arg) for arg in args]  # seq

    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
    #                          key=lambda x: -x[1])[:10]:
    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

    # print(xov_tmp.xovers)
    # assign parOrb to xov
    xov_tmp.parOrb_xy = [x.split('_')[0] for x in xov_tmp.xovers.filter(regex='^dR/[a-zA-Z0-9]+_.*$').columns.values]  # update partials list
    xov_tmp.parGlo_xy = [(a + b) for a in ['dR/'] for b in list(parGlo.keys())]
    xov_tmp.par_xy = xov_tmp.parOrb_xy+xov_tmp.parGlo_xy  # update partials list
    print("Parameters:",xov_tmp.parOrb_xy,xov_tmp.parGlo_xy,xov_tmp.par_xy)

    # update xovers table with LAT and LON
    xov_tmp = get_xov_latlon(xov_tmp, proj_df.loc[proj_df.partid == 'none'])

    xov_tmp.xovers.drop('xovid', axis=1).reset_index(inplace=True, drop=True)
    # print(xov_tmp.xovers.columns)
    pd.set_option('display.max_columns', 500)
    print(xov_tmp.xovers)

    end = time.time()

    print("Fine_xov finished after", end - start_finexov, "sec and located", len(xov_tmp.xovers),"out of previous", len(old_xovs),"xovers!")

    # Save to file
    if not os.path.exists(outdir + outdir_in + 'xov/'):
        os.mkdir(outdir + outdir_in + 'xov/')
    xov_tmp.save(outdir + outdir_in + 'xov/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '.pkl')  # one can split the df by trackA and save multiple pkl, one for each trackA if preferred

    print('Xov for ' + str(cmb) + ' processed and written to ' + outdir + outdir_in + 'xov/xov_' + str(cmb[0]) + '_' + str(
        cmb[1]) + '.pkl @' + time.strftime("%H:%M:%S", time.gmtime()))

    print("Process finished after ", end-start, "sec!")

    return xov_tmp


if __name__ == '__main__':
    start = time.time()

    cmb = [13, 14]

    xov_tmp = xov_prc_iters_run()

    end = time.time()

    print("Process finished after ", end - start, "sec!")