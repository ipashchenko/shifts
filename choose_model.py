import os
import math
from functools import partial
from sklearn import preprocessing
import nestle
import corner
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import george

data_dir = '/home/ilya/github/shifts/data'
data9 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp9.txt'),
                   usecols=[5, 7])
robust_scaler_t = preprocessing.RobustScaler()
robust_scaler_y = preprocessing.RobustScaler()
minmax_scaler_t = preprocessing.MinMaxScaler()
minmax_scaler_y = preprocessing.MinMaxScaler()
t = robust_scaler_t.fit_transform(data9[:, 0])
y = robust_scaler_y.fit_transform(data9[:, 1])
y_mm = minmax_scaler_y.fit_transform(data9[:, 1])
t_mm = minmax_scaler_t.fit_transform(data9[:, 0])
# plt.plot(data9[:, 0], data9[:, 1])
flux = np.loadtxt(os.path.join(data_dir, 'mojave_flux_core.txt'), usecols=[4,5])
t_flux = minmax_scaler_t.transform(flux[:, 0])
t_fine = np.linspace(0, 1, 500)

# p1 = np.polyfit(t, data9[:, 1], 1)
# p2 = np.polyfit(t, data9[:, 1], 2)

# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p1, data9[:, 0]), '.k')
# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p2, data9[:, 0]), '.k')
# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p1, data9[:, 0]))
# plt.plot(data9[:, 0], data9[:, 1] - np.polyval(p2, data9[:, 0]))
# plt.axhline(0)


def model(p, t):
    return np.polyval(p, t)


def lnlike(p, t, y):
    a, alpha, tau, d = np.exp(p[:4])
    gp = george.GP(a * george.kernels.RationalQuadraticKernel(alpha, tau) +
                   george.kernels.WhiteKernel(d))
    # gp = george.GP(a * george.kernels.ExpSquaredKernel(tau) +
    #                george.kernels.WhiteKernel(d))

    gp.compute(t, 0.0001)
    return gp.lnlikelihood(y - model(p[4:], t))


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

ndim = 7
# a, alpha, tau, d
ppfs = [_function_wrapper(sp.stats.uniform.ppf, [-10, 20], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-15, 20], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-15, 10], {}),
        # _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {}),
        _function_wrapper(sp.stats.norm.ppf, [0.0, 0.01], {}),
        _function_wrapper(sp.stats.norm.ppf, [1.0, 0.5], {}),
        # _function_wrapper(sp.stats.norm.ppf, [0.0, 0.1], {}),
        _function_wrapper(sp.stats.uniform.ppf, [-10, 20], {})]
hypercube = hypercube_partial(ppfs)
lnlikep = partial(lnlike, t=t_mm, y=y_mm)
result = nestle.sample(loglikelihood=lnlikep, prior_transform=hypercube,
                       ndim=ndim, npoints=100, method='multi',
                       callback=nestle.print_progress)
samples = nestle.resample_equal(result.samples, result.weights)
# Save re-weighted samples from posterior
# np.savetxt('samples.txt', samples)
fig = corner.corner(samples, show_titles=True, labels=ndim*['a'],
                    quantiles=[0.16, 0.5, 0.84], title_fmt='.3f')


p_mean = np.mean(samples, axis=0)
fig, axes = plt.subplots(1, 1)
axes.set_ylabel("Distance, mas")
axes.set_xlabel("Time, years")
t_fine_ = minmax_scaler_t.inverse_transform(t_fine)
for i, s in enumerate(samples[np.random.randint(len(samples), size=12)]):
    print("Plotting sample {}".format(i))
    # gp = george.GP(np.exp(s[0]) * george.kernels.ExpSquaredKernel(np.exp(s[1])))
    gp = george.GP(np.exp(s[0]) * george.kernels.RationalQuadraticKernel(np.exp(s[1]), np.exp(s[2])))
    gp.compute(t_mm, 0.001)
    m = gp.sample_conditional(y_mm - model(p_mean[4:], t_mm), t_fine)
    # axes.plot(t, m, color="#4682b4", alpha=0.25)
    axes.plot(t_fine_,
              minmax_scaler_y.inverse_transform(m + model(s[4:], t_fine)),
              color="green", alpha=0.25)
axes.plot(t_fine_,
          minmax_scaler_y.inverse_transform(model(p_mean[4:], t_fine)),
          color="red")
axes.plot(data9[:, 0], data9[:, 1], '.k')
plt.show()
# # fig.savefig(os.path.join("corner.png"), bbox_inches='tight', dpi=200)
