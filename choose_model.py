import os
import math
from functools import partial
import nestle
import corner
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import emcee
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, RationalQuadratic,
                                              RBF)
import george


data_dir = '/home/ilya/github/shifts/data'
data9 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp9.txt'),
                   usecols=[5, 7])
t_min = data9[0, 0]
data9[:, 0] -= t_min
# plt.plot(data9[:, 0], data9[:, 1])

t = np.linspace(0, math.ceil(data9[:, 0].max()), 300)

p1 = np.polyfit(data9[:, 0], data9[:, 1], 1)
p2 = np.polyfit(data9[:, 0], data9[:, 1], 2)

# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p1, data9[:, 0]), '.k')
# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p2, data9[:, 0]), '.k')
# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p1, data9[:, 0]))
# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p2, data9[:, 0]))
# plt.axhline(0)


def model(p, t):
    return np.polyval(p, t)


def lnlike(p, t, y):
    a, tau, d = np.exp(p[:3])
    # gp = george.GP(a * george.kernels.RationalQuadraticKernel(alpha, tau) +
    #                george.kernels.WhiteKernel(d))
    gp = george.GP(a * george.kernels.ExpSquaredKernel(tau) +
                   george.kernels.WhiteKernel(d))

    gp.compute(t, 0.001)
    return gp.lnlikelihood(y - model(p[3:], t))


class _function_wrapper(object):
    """
    This is a hack to make the likelihood function pickleable when ``args``
    and ``kwargs``are also included.
    """
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        try:
            return self.f(x, *self.args, **self.kwargs)
        except:
            import traceback
            print(" choose model: Exception while calling your prior pdf:")
            print(" params:", x)
            print(" args:", self.args)
            print(" kwargs:", self.kwargs)
            print(" exception:")
            traceback.print_exc()
            raise


# Setting up nestle sampler
def hypercube_full(u, ppfs):
    assert len(u) == len(ppfs)
    return [ppf(u_) for ppf, u_ in zip(ppfs, u)]


def hypercube_partial(ppfs):
    return partial(hypercube_full, ppfs=ppfs)

ndim = 5
# a, tau, d
ppfs = [_function_wrapper(sp.stats.uniform.ppf, [-10, 8], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-10, 7], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {}),
        # _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {})]
hypercube = hypercube_partial(ppfs)
lnlikep = partial(lnlike, t=data9[:, 0], y=data9[:, 1])
result = nestle.sample(loglikelihood=lnlikep, prior_transform=hypercube,
                       ndim=ndim, npoints=150, method='multi',
                       callback=nestle.print_progress)
samples = nestle.resample_equal(result.samples, result.weights)
# Save re-weighted samples from posterior
# np.savetxt('samples.txt', samples)
fig = corner.corner(samples, show_titles=True, labels=ndim*['a'],
                    quantiles=[0.16, 0.5, 0.84], title_fmt='.3f')

flux = np.loadtxt(os.path.join(data_dir, 'mojave_flux_core.txt'), usecols=[4,5])
flux[:, 0] -= t_min


fig, axes = plt.subplots(1, 1)
axes.set_ylabel("Distance, mas")
axes.set_xlabel("Time, years")
for i, s in enumerate(samples[np.random.randint(len(samples), size=24)]):
    print("Plotting sample {}".format(i))
    gp = george.GP(np.exp(s[0]) * george.kernels.ExpSquaredKernel(np.exp(s[1])))
    # gp = george.GP(np.exp(s[0]) * george.kernels.RationalQuadraticKernel(np.exp(s[1]), np.exp(s[2])))
    gp.compute(data9[:, 0], 0.001)
    m = gp.sample_conditional(data9[:, 1] - model(s[3:], data9[:, 0]), t)
    # axes.plot(t, m, color="#4682b4", alpha=0.25)
    axes.plot(t, m+model(s[3:], t), color="green", alpha=0.25)
p_mean = np.mean(samples, axis=0)
axes.plot(t, model(p_mean[3:], t), color="red")
axes.plot(data9[:, 0], data9[:, 1], '.k')
# # fig.savefig(os.path.join("corner.png"), bbox_inches='tight', dpi=200)
