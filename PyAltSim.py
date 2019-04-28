#!/usr/bin/env python3
# ----------------------------------
# PyXover
# ----------------------------------
# Author: Stefano Bertone
# Created: 16-Oct-2018
#
import warnings

from icrf2pbf import icrf2pbf
from setupROT import setupROT

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import shutil
import glob
import time
import pickleIO
import datetime

import numpy as np
import pandas as pd
from scipy.constants import c as clight
from scipy.interpolate import RectBivariateSpline
import multiprocessing as mp
import subprocess

import spiceypy as spice

# mylib
from prOpt import debug, parallel, outdir, auxdir, local, new_illumNG, apply_topo, vecopts
import astro_trans as astr
from ground_track import gtrack
from geolocate_altimetry import get_sc_ssb, get_sc_pla
import perlin2d

########################################
# start clock
start = time.time()

##############################################
class sim_gtrack(gtrack):
	def __init__(self, vecopts, orbID):
		gtrack.__init__(self,vecopts)
		self.orbID = orbID
		self.name = str(orbID)
		self.outdir = None

	def setup(self, df):
		df_ = df.copy()

		# get time of flight in ns from probe one-way range in km
		df_['TOF']=df_['rng']*2.*1.e3/clight
		# preparing df for geoloc
		df_['seqid'] = df_.index
		df_ = df_.rename(columns={"epo_tx": "ET_TX"})
		df_ = df_.reset_index(drop=True)
		# copy to self
		self.ladata_df = df_[['ET_TX', 'TOF','orbID', 'seqid']]

		# retrieve spice data for geoloc
		if not hasattr(self, 'SpObj'):
			# create interp for track
			self.interpolate()
		else:
			self.SpObj = pickleIO.load(auxdir + 'spaux_' + self.name + '.pkl')

		# actual processing
		self.lt_topo_corr(df=df_)
		self.setup_rdr()

	def lt_topo_corr(self, df, itmax=100, tol=1.e-2):
		"""
		iterate from a priori rough TOF @ ET_TX to account for light-time and
		terrain roughness and topography
		et, rng0 -> lon0, lat0, z0 (using geoloc)
		lon0, lat0 -> DEM elevation (using GMT), texture from stamp
		dz -> difference btw z0 from geoloc and "real" elevation z at lat0/lon0
		update range and tof -> new_rng = old_rng + dz

		:param df: table with tof, et (+all apriori data)
		:param itmax: max iters allowed
		:param tol: tolerance for convergence
		"""
		#self.ladata_df[["TOF"]] = self.ladata_df.loc[:,"TOF"] + 200./clight

		# a priori values for internal FULL df
		df.loc[:, 'converged'] = False
		df.loc[:, 'offnadir'] = 0

		for it in range(itmax):

			# tof and rng from previous iter as input for new geoloc
			old_tof = self.ladata_df.loc[:, 'TOF'].values
			rng_apr = old_tof*clight/2.

			# read just lat, lon, elev from geoloc (reads ET and TOF and updates LON, LAT, R in df)
			self.geoloc()
			lontmp, lattmp, rtmp = np.transpose(self.ladata_df[['LON','LAT','R']].values)
			r_bc = rtmp + vecopts['PLANETRADIUS']*1.e3

			# use lon and lat to get "real" elevation from map
			radius = self.get_topoelev(lattmp, lontmp)

			# use "real" elevation to get bounce point coordinates
			bcxyz_pbf = astr.sph2cart(radius,lattmp,lontmp)
			bcxyz_pbf = np.transpose(np.vstack(bcxyz_pbf))

			# get S/C body fixed position (useful to update ranges, has to be computed on reduced df)
			scxyz_tx_pbf = self.get_sc_pos_bf(self.ladata_df)
			# compute range btw probe@TX and bounce point@BC (no BC epoch needed, all coord planet fixed)
			rngvec = (bcxyz_pbf - scxyz_tx_pbf)
			# compute correction for off-nadir observation
			offndr = np.arccos(np.einsum('ij,ij->i', rngvec, -scxyz_tx_pbf) /
							   np.linalg.norm(rngvec, axis=1) /
							   np.linalg.norm(scxyz_tx_pbf, axis=1))

			# compute residual between "real" elevation and geoloc (based on a priori TOF)
			dr = (r_bc - radius) * np.cos(offndr)

			# update range
			rng_new = rng_apr + dr

			# update tof
			tof = 2.*rng_new/clight
			self.ladata_df.loc[:, 'TOF'] = tof  # convert to update
			self.ladata_df.loc[:, 'converged'] = abs(dr) < tol
			self.ladata_df.loc[:, 'offnadir'] = np.rad2deg(offndr)

			if it == 0:
				df = self.ladata_df.copy()
			else:
				df.update(self.ladata_df)

			if debug:
				print("it = "+str(it))
				print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol))

			if (max(abs(dr)) < tol):
				#print("Convergence reached")
				# pass all epochs to next step
				self.ladata_df = df.copy()
				break
			elif (it == itmax - 1):
				print('### altsim: Max number of iterations reached!')
				print("it = "+str(it))
				print("max resid:", max(abs(dr)), "# > tol:", np.count_nonzero(abs(dr) > tol))
				print('offnadir max',max(np.rad2deg(offndr)))
				self.ladata_df = df.copy()  # keep non converged but set chn>5 (bad msrmts)
				break
			else:
				# update global df used in geoloc at next iteration (TOF)
				#df = df[df.loc[:, 'offnadir'] < 5]
				# only operate on non-converged epochs for next iteration
				self.ladata_df = df[df.loc[:, 'converged'] == False].copy()
				# self.ladata_df = df.copy()

	def get_topoelev(self, lattmp, lontmp):

		if apply_topo:
			# st = time.time()
			gmt = 1
			if gmt:
				gmt_in = 'gmt_'+self.name+'.in'
				if os.path.exists('tmp/'+gmt_in):
					os.remove('tmp/'+gmt_in)
				np.savetxt('tmp/'+gmt_in, list(zip(lontmp, lattmp)))
				dem = '/att/nobackup/emazaric/MESSENGER/data/GDR/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
				#'MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_32ppd_HgM008frame.GRD'
				if local == 0:
					r_dem = subprocess.check_output(
						['grdtrack', gmt_in,
						 '-G'+dem],
						universal_newlines=True, cwd='tmp')
					r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
					# np.savetxt('gmt_'+self.name+'.out', r_dem)
				else:
					# r_dem = np.loadtxt('tmp/gmt_' + self.name + '.out')
					r_dem = subprocess.check_output(
						['grdtrack', gmt_in, '-G'+dem],
						universal_newlines=True, cwd='tmp')
					r_dem = np.fromstring(r_dem, sep=' ').reshape(-1, 3)[:, 2]
					
			# else:
			#     print(lontmp)
			#     print(lontmp+180.)
			#     print(lattmp)
			#     r_dem = self.dem_xr.interp(lon=lontmp+180.,lat=lattmp).to_dataframe().loc[:, 'z'].values

			# print(r_dem)
			# en = time.time()
			# print(en-st)
			# exit()

			texture_noise = self.apply_texture(np.mod(lattmp, 0.25), np.mod(lontmp, 0.25), grid=False)
			# print("texture noise check",texture_noise,r_dem,rtmp)

			# update Rmerc with r_dem/text
			radius = vecopts['PLANETRADIUS'] * 1.e3 + r_dem + texture_noise
			# print(radius,r_dem,texture_noise)
		else:
			radius = vecopts['PLANETRADIUS'] * 1.e3

		return radius

	def get_sc_pos_bf(self, df):
		et_tx = df.loc[:, 'ET_TX'].values
		sc_pos, sc_vel = get_sc_ssb(et_tx, self.SpObj, self.pertPar, self.vecopts)
		scpos_tx_p, _ = get_sc_pla(et_tx, sc_pos, sc_vel, self.SpObj, self.vecopts)
		rotpar, upd_rotpar = setupROT(self.pertPar['dRA'], self.pertPar['dDEC'], self.pertPar['dPM'],
									  self.pertPar['dL'])
		tsipm = icrf2pbf(et_tx, upd_rotpar)
		scxyz_tx_pbf = np.vstack([np.dot(tsipm[i], scpos_tx_p[i]) for i in range(0, np.size(scpos_tx_p, 0))])
		return scxyz_tx_pbf

	def setup_rdr(self):
		df_ = self.ladata_df.copy()

		# only select nadir data
		#df_ = df_[df_.loc[:,'offnadir']<5]

		mlardr_cols = ['geoc_long', 'geoc_lat', 'altitude', 'EphemerisTime', 'MET', 'frm',
					   'chn', 'Pulswd', 'thrsh', 'gain', '1way_range', 'Emiss', 'TXmJ',
					   'UTC', 'TOF_ns_ET', 'Sat_long', 'Sat_lat', 'Sat_alt', 'Offnad', 'Phase',
					   'Sol_inc', 'SCRNGE', 'seqid']
		self.rdr_df = pd.DataFrame(columns=mlardr_cols)

		# assign "bad chn" to non converged observations
		df_['chn']= 0
		df_.loc[df_['converged'] == False, 'chn'] = 10

		# update other columns for compatibility with real data format
		df_['TOF_ns_ET'] = np.round(df_['TOF'].values * 1.e9, 10)
		df_['UTC']= pd.to_datetime(df_['ET_TX'], unit='s',
				   origin=pd.Timestamp('2000-01-01T12:00:00'))

		df_ = df_.rename(columns={'ET_TX':'EphemerisTime',
								  'LON':'geoc_long','LAT':'geoc_lat','R':'altitude',
								  })
		df_ = df_.reset_index(drop=True)
		if local:
			self.rdr_df = self.rdr_df.append(df_[['EphemerisTime','geoc_long','geoc_lat','altitude',
										 'UTC', 'TOF_ns_ET','chn','seqid']])[mlardr_cols]
		else:
			self.rdr_df = self.rdr_df.append(df_[['EphemerisTime','geoc_long','geoc_lat','altitude',
										 'UTC', 'TOF_ns_ET','chn','seqid']], sort=True)[mlardr_cols]

##############################################

def prepro_ilmNG(illumNGf):

	li = []
	for f in illumNGf:
		print("Processing", f)
		df = pd.read_csv(f, index_col=None, header=0, names=[f.split('.')[-1]])
		li.append(df)

	#df_ = dfin.copy()
	df_ = pd.concat(li, axis=1)
	df_=df_.apply(pd.to_numeric, errors='coerce')
	#print(df_.rng.min())
	
	df_ = df_[df_.rng < 1600]
	df_=df_.rename(columns={"xyzd": "epo_tx"})
	#print(df_.dtypes)

	df_['diff'] = df_.epo_tx.diff().fillna(0)
	#print(df_[df_['diff'] > 1].index.values)
	arcbnd = [df_.index.min()]
	# new arc if observations separated by more than 1h
	arcbnd.extend(df_[df_['diff'] > 3600].index.values)
	arcbnd.extend([df_.index.max() + 1])
	#print(arcbnd)
	df_['orbID'] = 0
	for i,j in zip(arcbnd,arcbnd[1:]):
	     orbid = (datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(seconds=df_.loc[i, 'epo_tx'])).strftime("%y%m%d%H%M")
	     df_.loc[df_.index.isin(np.arange(i, j)), 'orbID'] = orbid

	return df_

def sim_track(args):
	track, df, i, outdir_ = args
	#print(track.name)
	if os.path.isfile(outdir_+'MLASIMRDR'+track.name+'.TAB') == False:
		track.setup(df[df['orbID']==i])
		track.rdr_df.to_csv(outdir_+'MLASIMRDR'+track.name+'.TAB', index=False, sep=',',na_rep='NaN')
		print('Simulated observations written to',outdir_+'MLASIMRDR'+track.name+'.TAB')
	else:
		print('Simulated observations ',outdir_+'MLASIMRDR'+track.name+'.TAB already exists. Skip.')

def main(arg): #dirnam_in = 'tst', ampl_in=35,res_in=0):

	ampl_in = list(arg)[0]
	res_in = list(arg)[1]
	dirnam_in = list(arg)[2]
	epos_in = list(arg)[3]

	print('dirnam_in', dirnam_in)
	print('epos_in', epos_in)

	if local == 0:
		data_pth = '/att/nobackup/sberton2/MLA/MLA_RDR/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
		dataset = ''  # 'small_test/' #'test1/' #'1301/' #
		data_pth += dataset
		# load kernels
		spice.furnsh('/att/nobackup/emazaric/MESSENGER/data/furnsh/furnsh.MESSENGER.def')  # 'aux/mymeta')
	else:
		#data_pth = '/home/sberton2/Works/NASA/Mercury_tides/data/'  # /home/sberton2/Works/NASA/Mercury_tides/data/'
		#dataset = "test/"  # ''  # 'small_test/' #'1301/' #
		#data_pth += dataset
		# load kernels
		spice.furnsh(auxdir + 'mymeta')  # 'aux/mymeta')

	if parallel:
		# set ncores
		ncores = mp.cpu_count() - 1  # 8
		print('Process launched on ' + str(ncores) + ' CPUs')

	#out = spice.getfov(vecopts['INSTID'][0], 1)
	# updated w.r.t. SPICE from Mike's scicdr2mat.m
	vecopts['ALTIM_BORESIGHT'] = [0.0022105, 0.0029215, 0.9999932892]  # out[2]
	###########################

	# generate list of epochs
	if new_illumNG and True:
		# read all MLA datafiles (*.TAB in data_pth) corresponding to the given time period
		allFiles = glob.glob(os.path.join(data_pth, 'MLAS??RDR' + epos_in + '*.TAB'))
		print(allFiles)

		# Prepare list of tracks
		tracknames = ['gtrack_' + fil.split('.')[0][-10:] for fil in allFiles]
		epo_in = []
		for track_id, infil in zip(tracknames, allFiles):
			track = track_id
			track = gtrack(vecopts)
			track.prepro(infil)
			epo_in.extend(track.ladata_df.ET_TX.values)

		epo_in = np.sort(np.array(epo_in))
		#print(epo_in)
		#print(epo_in.shape)
		#print(np.sort(epo_in)[0],np.sort(epo_in)[-1])
		#print(np.sort(epo_in)[-1])

	else:
		epo0 = 410270400  # get as input parameter
		# epo_tx = np.array([epo0+i for i in range(86400*7)])
		subpnts = 10
		epo_tx = np.array([epo0 + i / subpnts for i in range(86400 * subpnts)])

	# pass to illumNG
	if local:
		if new_illumNG:
			np.savetxt("tmp/epo_mla_" + epos_in + ".in", epo_tx, fmt="%4d")
			print("Do you have all of illumNG predictions?")
			exit()
		path = '../aux/illumNG/sph_7d_mla/'  # _1s/'  #  sph/' #grd/' # use your path
		illumNGf = glob.glob(path + "bore*")
	else:
		if new_illumNG:
			np.savetxt("tmp/epo_mla_" + epos_in + ".in", epo_in, fmt="%10.5f")
			print("illumNG call")
			exit()
			if not os.path.exists("illumNG/"):
				print('*** create and copy required files to ./illumNG')
				exit()

			shutil.copy("tmp/epo_mla_"+epos_in+".in",'../_MLA_Stefano/epo.in')
			illumNG_call = subprocess.call(
				['sbatch', 'doslurmEM', 'MLA_raytraces.cfg'],
				universal_newlines=True, cwd="../_MLA_Stefano/") #illumNG/")
			for f in glob.glob("../_MLA_Stefano/bore*"):
				shutil.move(f, auxdir+'/illumNG/grd/'+epos_in+"_"+f.split('/')[1])

		path = auxdir+'illumng/mlatimes_'+epos_in+'/' #sph/' # use your path
		print('illumng dir', path)
		illumNGf = glob.glob(path + "/bore*")

	#else:
	# launch illumNG directly
	df = prepro_ilmNG(illumNGf)
	print('illumNGf',illumNGf)

	if apply_topo:
		# read and interpolate DEM
		# open netCDF file
		# nc_file = "/home/sberton2/Works/NASA/Mercury_tides/MSGR_DEM_USG_SC_I_V02_rescaledKM_ref2440km_4ppd_HgM008frame.GRD"
		# sim_gtrack.dem_xr = xr.open_dataset(nc_file)

		#prepare surface texture "stamp" and assign the interpolated function as class attribute
		np.random.seed(62)
		shape_text = 1024
		res_text = 2**res_in
		depth_text = 5
		size_stamp = 0.25
		amplitude = ampl_in
		noise = perlin2d.generate_periodic_fractal_noise_2d(amplitude, (shape_text, shape_text), (res_text, res_text), depth_text)
		interp_spline = RectBivariateSpline(np.array(range(shape_text)) / shape_text * size_stamp,
											np.array(range(shape_text)) / shape_text * size_stamp,
											noise)
		sim_gtrack.apply_texture = interp_spline

	# Process tracks
	# tracks = []
	# for i in list(df.groupby('orbID').groups.keys()):
	#     if debug:
	#         print("Processing",i)
	#     tracks.append(sim_gtrack(vecopts, i))
	#
	# print(tracks)
	# print([tr.name for tr in tracks])

	if local:
	  outdir_ = outdir + dirnam_in
	else:
	  outdir_ = dirnam_in

	print(outdir_)
	if not os.path.exists(outdir_):
		os.makedirs(outdir_, exist_ok=True)

	# loop over all gtracks
	print('orbs',list(df.groupby('orbID').groups.keys()))
	args = ((sim_gtrack(vecopts, i), df, i, outdir_) for i in list(df.groupby('orbID').groups.keys()))

	if parallel and False: # incompatible with grdtrack call ...
		# print((mp.cpu_count() - 1))
		pool = mp.Pool(processes=ncores)  # mp.cpu_count())
		_ = pool.map(sim_track, args)  # parallel
		pool.close()
		pool.join()
	else:
		_ = [sim_track(arg) for arg in args]  # seq


##############################################
if __name__ == '__main__':

	import sys

	##############################################
	# launch program and clock
	# -----------------------------
	start = time.time()

	if len(sys.argv)==1:

		args = sys.argv[0]

		main(args)
	else:
		print("PyAltSim running with standard args...")
		main()

	# stop clock and print runtime
	# -----------------------------
	end = time.time()
	print('----- Runtime = ' + str(end - start) + ' sec -----' + str((end - start) / 60.) + ' min -----')
