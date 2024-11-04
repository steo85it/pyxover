#!/usr/bin/env python3
# ---------------------------------------------------------
# Generate fractal noise appropriate for terrain simulation
# ---------------------------------------------------------
# Author: Stefano Bertone
# Created: 16-May-2019
#
# source: https://github.com/pvigier/perlin-numpy
# check if similar to grdinfo mla01:/Volumes/disk1/mazarico/CLIPPER/scripts/Matlab/Topography/gaussii.grd
# Command to create: grdfft -I gaussi.grd -Ggaussii.grd
# Command to cut: grdcut -R2844/5796/2844/5796 gaussii16384_8640cut.grd -Ggaussii16384_2952cut.grd

import numpy as np
import xarray as xr

# def generate_perlin_noise_2d(shape, res):
#     def f(t):
#         return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
#
#     delta = (res[0] / shape[0], res[1] / shape[1])
#     d = (shape[0] // res[0], shape[1] // res[1])
#     grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
#     # Gradients
#     angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
#     gradients = np.dstack((np.cos(angles), np.sin(angles)))
#     g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
#     g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
#     g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
#     g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
#     # Ramps
#     n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
#     n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
#     n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
#     n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
#     # Interpolation
#     t = f(grid)
#     n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
#     n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
#     return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    # Make the noise tileable
    gradients[-1, :] = gradients[0, :]
    gradients[:, -1] = gradients[:, 0]
    # Same as before
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    # print(np.array(res)*octaves,shape,np.mod(np.array(res)*octaves,shape))
    # if np.shape(res*octaves)[0]%np.shape(shape)[0] == 0:
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    # else:
    #    print("shape must be a multiple of octaves*res.")
    #    exit(2)

    return noise


def generate_periodic_fractal_noise_2d(amplitude, shape, res, octaves=1, persistence=0.5):
    noise = generate_fractal_noise_2d(shape, res, octaves, persistence)
    # print("pre",noise)
    noise *= amplitude
    # print("post",noise)
    # _ = np.hstack([noise,np.flip(noise,axis=1)])
    # noise = np.vstack([_,np.flip(_,axis=0)])
    return noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # from scipy.interpolate import RectBivariateSpline
    from hillshade import hill_shade
    import seaborn as sns

    # np.random.seed(0)
    # noise = generate_perlin_noise_2d((256, 256), (8, 8))
    # plt.imshow(noise, cmap='gray', interpolation='lanczos')
    # plt.colorbar()

    amplitude = 20
    resolution = 3

    np.random.seed(62)
    shape_text = 1024
    res_text = 2 ** resolution
    depth_text = 5
    size_stamp = 0.25
    noise = generate_periodic_fractal_noise_2d(amplitude, (shape_text, shape_text), (res_text, res_text),
                                               depth_text,  persistence=0.65)
    print(noise.shape)
    noise_xr = xr.DataArray(noise,
                           coords={'y': np.linspace(0, 0.25, noise.shape[1]),
                                   'x': np.linspace(0, 0.25, noise.shape[0])},
                           dims=["y", "x"])
    noise_xr.rio.write_crs("+proj=lonlat +units=m +a=1737.4e3 +b=1737.4e3 +no_defs", inplace=True)
    print(noise_xr)
    noise_xr.plot()
    plt.show()

    noise_xr.rio.to_raster('/home/sberton2/Scaricati/test_fract_noise_res1.tif')



    #
    # noise = hill_shade(noise, terrain=noise * 10)
    # # noise = abs(noise)**2.1/(abs(noise)**2.1).max()*35
    # # interp_spline = RectBivariateSpline(np.array(range(shape_text * 2)) / shape_text * 2. * size_stamp,
    # #                                    np.array(range(shape_text * 2)) / shape_text * 2. * size_stamp,
    # #                                    noise)
    # #
    # sns.set_context('notebook')
    # plt.figure()
    # ax = plt.imshow(noise, cmap='viridis', interpolation='None', extent=(0, 0.25, 0, 0.25))
    #
    # plt.xlabel('degrees')
    # plt.ylabel('degrees')
    # plt.title('Simulated small-scale topography', pad=10)
    #
    # cbar = plt.colorbar()
    # cbar.set_label('elevation (m)')
    # cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
    # cbar.set_ticklabels([-40, -20, 0, 20, 40])
    #
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig("/home/sberton2/Scaricati/test_fract_noise_res1.pdf")
