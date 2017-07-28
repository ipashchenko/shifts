import os
import numpy as np
import matplotlib.pyplot as plt
import emcee
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, RationalQuadratic,
                                              RBF)



class MyGaussianProcessRegressor(GaussianProcessRegressor):
    def lnlikelihood(self, theta, y=None):
        # Update data if supplied
        if y is not None:
            self.y_train_ = y

        return self.log_marginal_likelihood(theta=theta)


data_dir = '/home/ilya/github/shifts/data'
data9 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp9.txt'),
                   usecols=[5, 7])
data12 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp12.txt'),
                    usecols=[5, 7])
data21 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp21.txt'),
                    usecols=[5, 7])
data26 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp26.txt'),
                    usecols=[5, 7])
t_min = data9[0, 0]


for data in (data9, data12, data21, data26):
    data[:, 0] -= t_min
    plt.plot(data[:, 0], data[:, 1])

t = np.linspace(0, 12, 300)

p9 = np.polyfit(data9[:, 0], data9[:, 1], 2)
p12 = np.polyfit(data12[:, 0], data12[:, 1], 2)
p21 = np.polyfit(data21[:, 0], data21[:, 1], 2)
p26 = np.polyfit(data26[:, 0], data26[:, 1], 2)

for data, p in zip((data9, data12, data21, data26), (p9, p12, p21, p26)):
    plt.plot(data[:, 0], data[:, 1], '.k')
    plt.plot(t, np.polyval(p, t))


def model2(p, t):
    return np.polyval(p, t)


p0 = [1., 2., 0.1, 0.1]
k1 = p0[0] * RationalQuadratic(length_scale=p0[1], alpha=p0[2])
k2 = WhiteKernel(noise_level=p0[3] ** 2,
                 noise_level_bounds=(1e-4, np.inf))
kernel = k1 + k2


gp9 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                               alpha=(0.1 / data9[:, 1]) ** 2,
                               copy_X_train=False)
# Make ``y`` dependent on parameters ``p``
X9 = np.atleast_2d(data9[:, 0]).T
y9 = data9[:, 1] - model2(p9, data9[:, 0])
gp9.fit(X9, y9)

gp12 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                alpha=(0.1 / data12[:, 1]) ** 2)
X12 = np.atleast_2d(data12[:, 0]).T
y12 = data12[:, 1] - model2(p12, data12[:, 0])
gp12.fit(X12, y12)

gp21 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                alpha=(0.1 / data21[:, 1]) ** 2)
X21 = np.atleast_2d(data21[:, 0]).T
y21 = data21[:, 1] - model2(p21, data21[:, 0])
gp21.fit(X21, y21)

gp26 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                alpha=(0.1 / data26[:, 1]) ** 2)
X26 = np.atleast_2d(data26[:, 0]).T
y26 = data26[:, 1] - model2(p26, data26[:, 0])
gp26.fit(X26, y26)

p0 = gp9.kernel_.theta


def lnlike(p):
    # Update yi-th
    global y9
    global y12
    global y21
    global y26
    y9 = data9[:, 1] - model2(p[4:7], data9[:, 0])
    y12 = data12[:, 1] - model2(p[7:10], data12[:, 0])
    y21 = data21[:, 1] - model2(p[10:13], data21[:, 0])
    y26 = data26[:, 1] - model2(p[13:16], data26[:, 0])
    return (gp9.log_marginal_likelihood(p[:4]) +
            gp12.log_marginal_likelihood(p[:4]) +
            gp21.log_marginal_likelihood(p[:4]) +
            gp26.log_marginal_likelihood(p[:4])).sum()


def lnprior(p):
    # Prior on GP hyper parameters
    if not -10 < p[0] < 10:
        return -np.inf
    if not -10 < p[1] < 10:
        return -np.inf
    if not -10 < p[2] < 10:
        return -np.inf
    if not -10 < p[3] < 10:
        return -np.inf
    # Prior on p9
    if not -0.03 < p[4] < 0.03:
        return -np.inf
    if not 0.4 < p[5] < 1.2:
        return -np.inf
    if not 0.4 < p[6] < 1.2:
        return -np.inf
    # Prior on p12
    if not -0.003 < p[7] < 0.003:
        return -np.inf
    if not 0.4 < p[8] < 1.4:
        return -np.inf
    if not -5 < p[9] < 0:
        return -np.inf
    # Prior on p21
    if not -0.012 < p[10] < 0.012:
        return -np.inf
    if not 0.2 < p[11] < 0.6:
        return -np.inf
    if not 0.0 < p[12] < 3.0:
        return -np.inf
    # Prior on p26
    if not -0.006 < p[13] < 0.006:
        return -np.inf
    if not 0.4 < p[14] < 1.2:
        return -np.inf
    if not 3.0 < p[15] < 10:
        return -np.inf
    return 0.0


def lnprob(p):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lnlike(p) + lp


nwalkers = 80
ndim = 16
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
p0 = emcee.utils.sample_ball(list(p0) + list(p9) + list(p12) + list(p21) +
                             list(p26), [0.1, 0.1, 0.1, 0.1] +
                             4*[0.001, 0.02, 0.1],
                             size=nwalkers)
p0, lnp, _ = sampler.run_mcmc(p0, 100)
sampler.reset()
p0, lnp, _ = sampler.run_mcmc(p0, 200)

p_map = p0[np.argmax(lnp)]
p_map[:4] = np.exp(p_map[:4])


k1 = p_map[0] * RationalQuadratic(length_scale=p_map[1], alpha=p_map[2])
k2 = WhiteKernel(noise_level=p_map[3] ** 2,
                 noise_level_bounds=(1e-4, np.inf))
kernel = k1 + k2
gp9 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                              alpha=(0.1 / data9[:, 1]) ** 2)
gp9.fit(np.atleast_2d(data9[:, 0]).T, data9[:, 1] - model2(p_map[4:7],
                                                           data9[:, 0]))

gp12 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                              alpha=(0.1 / data12[:, 1]) ** 2)
gp12.fit(np.atleast_2d(data12[:, 0]).T, data12[:, 1] - model2(p_map[7:10],
                                                              data12[:, 0]))

gp21 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                              alpha=(0.1 / data21[:, 1]) ** 2)
gp21.fit(np.atleast_2d(data21[:, 0]).T, data21[:, 1] - model2(p_map[10:13],
                                                              data21[:, 0]))

gp26 = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                              alpha=(0.1 / data26[:, 1]) ** 2)
gp26.fit(np.atleast_2d(data26[:, 0]).T, data26[:, 1] - model2(p_map[13:16],
                                                              data26[:, 0]))


for data, p, gp in zip((data9, data12, data21, data26), (p_map[4:7],
                                                         p_map[7:10],
                                                         p_map[10:13],
                                                         p_map[13:16]),
                       (gp9, gp12, gp21, gp26)):
    plt.plot(data[:, 0], data[:, 1], '.')
    y = gp.predict(np.atleast_2d(t).T)
    plt.plot(t, np.polyval(p, t) + y)
