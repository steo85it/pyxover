import numpy as np


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
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
    #print(np.array(res)*octaves,shape,np.mod(np.array(res)*octaves,shape))
    #if np.shape(res*octaves)[0]%np.shape(shape)[0] == 0:
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    #else:
    #    print("shape must be a multiple of octaves*res.")
    #    exit(2)

    return noise

def generate_periodic_fractal_noise_2d(amplitude, shape, res, octaves=1, persistence=0.5):

    noise = generate_fractal_noise_2d(shape, res, octaves, persistence)
    noise *= amplitude
    _ = np.hstack([noise,np.flip(noise,axis=1)])
    noise = np.vstack([_,np.flip(_,axis=0)])
    return noise

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.interpolate import RectBivariateSpline


    #np.random.seed(0)
    #noise = generate_perlin_noise_2d((256, 256), (8, 8))
    #plt.imshow(noise, cmap='gray', interpolation='lanczos')
    #plt.colorbar()

    np.random.seed(0)
    noise = generate_periodic_fractal_noise_2d(5, (2560, 2560), (16, 16), 5)
    interp_spline = RectBivariateSpline(np.array(range(5120))/5120.*0.25,
                                        np.array(range(5120)) / 5120. * 0.25,
                                        noise)

    #print(interp_spline(0.125,0.125))
    #print(noise[2560,2560])

    #print(noise)
    #print(np.flip(noise,axis=1))
    #
    plt.figure()
    plt.imshow(noise, cmap='gray', interpolation='lanczos')
    plt.colorbar()
    plt.savefig("test_fract_noise.pdf")