#!/usr/bin/env python3


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from pyaltsim import perlin2d

res_in = 3
ampl_in = 20
np.random.seed(62)
shape_text = 1024
res_text = 2 ** res_in
depth_text = 5
size_stamp = 0.25
amplitude = ampl_in
noise = perlin2d.generate_periodic_fractal_noise_2d(amplitude, (shape_text, shape_text), (res_text, res_text),
depth_text, persistence=0.65)
interp_spline = RectBivariateSpline(np.array(range(shape_text)) / shape_text * size_stamp,
                                    np.array(range(shape_text)) / shape_text * size_stamp,
                                    noise)


  
# convert array into dataframe 
DF = pd.DataFrame(noise) 
  
# save the dataframe as a csv file 
DF.to_csv("data1.csv")




plt.figure()
plt.imshow(noise, cmap='gray', interpolation='lanczos')
plt.colorbar()
plt.show()

if False:
   plt.figure()
   # lat = np.linspace(-90,90,200)
   lat = np.linspace(0,0.25,10000)
   lon = [60 for i in lat]
   Rm = 2410
   alt = 200
   GM = 7179.292
   v = np.sqrt(GM/(Rm+alt)) # [km/s]
   f_a = 10
   bs = v/f_a/2 # [km]
   texture_noise = interp_spline(np.mod(lat, 0.25), np.mod(lon, 0.25), grid=False)
   texture_noise2 = interp_spline(np.mod(lat+bs/Rm*180/np.pi, 0.25), np.mod(lon, 0.25), grid=False)

   slope = np.arctan(np.diff(texture_noise)/min(np.diff(lat))*180/np.pi/Rm/1000)*180/np.pi
   plt.plot(texture_noise, lat)
   plt.figure()
   slope = np.arctan((texture_noise-texture_noise2)/bs/1000)*180/np.pi
   plt.plot(slope)
   plt.figure()
   plt.hist(slope,bins=50)
   print(sum(abs(slope)<8)/len(slope))
   print(sum(abs(slope)<3.5)/len(slope))
   print(np.mean(abs(slope)))
   print(np.std(slope))
   # plt.plot(texture_noise,lat*np.pi/180*Rm)
   amplitude = ampl_in/4
   # res_text = 2 ** (res_in-1)
   noise = perlin2d.generate_periodic_fractal_noise_2d(amplitude, (shape_text, shape_text), (res_text, res_text),
   depth_text, persistence=0.65)
   interp_spline = RectBivariateSpline(np.array(range(shape_text)) / shape_text * size_stamp,
                                    np.array(range(shape_text)) / shape_text * size_stamp,
                                    noise)
   texture_noise = interp_spline(np.mod(lat, 0.25), np.mod(lon, 0.25), grid=False)
   # plt.plot(texture_noise,lat*np.pi/180*Rm)
   plt.show()
   plt.savefig('noise_lat_' + str(res_in) + 'res_' + str(amplitude) + 'amp.png')
   # plt.show()



        