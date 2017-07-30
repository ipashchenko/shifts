import os
import numpy as np
from scipy.stats import uniform
from functools import partial
import matplotlib.pyplot as plt
import george
import nestle
import corner


data_dir = '/home/ilya/github/shifts/data'
data9 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp9.txt'),
                   usecols=[5, 7])
data12 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp12.txt'),
                    usecols=[5, 7])
data21 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp21.txt'),
                    usecols=[5, 7])
data26 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp26.txt'),
                    usecols=[5, 7])
data_set = (data9, data12, data21, data26)
component_names = ("9", "12", "21", "26")
flux = np.loadtxt(os.path.join(data_dir, 'mojave_flux_core.txt'), usecols=[4,5])
t_min = np.min([data[0, 0] for data in data_set])
flux[:, 0] -= t_min

for data, label in zip(data_set, component_names):
    data[:, 0] -= t_min
#     plt.plot(data[:, 0], data[:, 1], label=label)
#     plt.plot(data[:, 0], data[:, 1], '.k')
# plt.legend(loc='best')
# plt.xlabel("Time from {}, years".format(t_min))
# plt.ylabel("Distance from core, mas")
# plt.tight_layout()
# plt.show()

t_max = np.max([data[-1, 0] for data in data_set])
t_fine = np.linspace(0, t_max, 500)


def model(p, t):
    """
    Model of the trend in single component motion.

    :param p:
        Iterable of polynomial coefficient.
    :param t:
        Time from the common start time.
    """
    return np.polyval(p, t)


def lnlike_single(p, t, y):
    """
    Log of likelihood for observed kinematics of single component.

    :param p:
        Iterable of the parameters.
    :param t:
        Time.
    :param y:
        Observed positions of component.
    :return:
        Log of likelihood.

    :notes:
        Model consists of per-component trend + GP & common to other components
        GP.
    """
    a_common, tau_common = np.exp(p[:2])
    a, tau, d = np.exp(p[2:5])
    # gp = george.GP(a * george.kernels.RationalQuadraticKernel(alpha, tau) +
    #                george.kernels.WhiteKernel(d))
    gp = george.GP(a_common * george.kernels.ExpSquaredKernel(tau_common) +
                   a * george.kernels.ExpSquaredKernel(tau) +
                   george.kernels.WhiteKernel(d))

    gp.compute(t, 0.001)
    return gp.lnlikelihood(y - model(p[5:], t))


def lnlike_common(p, data_set, models_dims):
    """
    Log of likelihood for observed kinematics of single component.

    :param p:
        Iterable of the parameters.
    :param data_set:
        Iterable of 2D numpy arrays with time and positions for each components.
    :param models_dims:
        Iterable of component specific number of parameters.
    :return:
        Log of likelihood.
    """
    p_common = p[:2]
    p_others = p[2:]
    j = 0
    result = list()
    for i, data in enumerate(data_set):
        t, y = data[:, 0], data[:, 1]
        # Adding 3 means per-component SE+WN
        p_component = list(p_common) + list(p_others[j: j+3+models_dims[i]])
        j += 3+models_dims[i]
        result.append(lnlike_single(p_component, t, y))

    return sum(result)


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

# 2 means linear
models_dims = (2, 2, 2, 2)
ndim = 2 + 3*len(models_dims) + sum(models_dims)

# Priors on common GP parameters
ppfs = [_function_wrapper(uniform.ppf, [-10, 8], {}),
        _function_wrapper(uniform.ppf, [-10, 9], {})]

# Component specific parameters priors
for model_dim in models_dims:
    # For each component long-scale SE + independent white noise
    ppfs += [_function_wrapper(uniform.ppf, [-10, 20], {}),
             _function_wrapper(uniform.ppf, [-1, 10], {}),
             _function_wrapper(uniform.ppf, [-10, 10], {})]
    ppfs += [_function_wrapper(uniform.ppf, [-10, 20], {}) for i in
             range(model_dim)]

hypercube = hypercube_partial(ppfs)
lnlike = partial(lnlike_common, data_set=data_set, models_dims=models_dims)
result = nestle.sample(loglikelihood=lnlike, prior_transform=hypercube,
                       ndim=ndim, npoints=50, method='multi',
                       callback=nestle.print_progress)

samples = nestle.resample_equal(result.samples, result.weights)
# Save re-weighted samples from posterior
# np.savetxt('samples.txt', samples)
fig = corner.corner(samples, show_titles=True, labels=ndim*['a'],
                    quantiles=[0.16, 0.5, 0.84], title_fmt='.3f')
fig.show()

component_colors = ('r', 'b', 'g', 'y')
# fig, axes = plt.subplots(1, 1)
# axes.set_ylabel("Distance, mas")
# axes.set_xlabel("Time, years")
# for i, s in enumerate(samples[np.random.randint(len(samples), size=6)]):
#     print("Plotting sample {}".format(i))
#     s_common = s[:3]
#     s_others = s[3:]
#     j = 0
#     for i, data in enumerate(data_set):
#         t, y = data[:, 0], data[:, 1]
#         s_component = list(s_common) + list(s_others[j: j+models_dims[i]])
#         j += models_dims[i]
#         gp = george.GP(np.exp(s_component[0]) *
#                        george.kernels.ExpSquaredKernel(np.exp(s_component[1])))
#         # gp = george.GP(np.exp(s[0]) * george.kernels.RationalQuadraticKernel(np.exp(s[1]), np.exp(s[2])))
#         gp.compute(t, 0.001)
#         m = gp.sample_conditional(y - model(s_component[3:], t), t_fine)
#         # axes.plot(t, m, color="#4682b4", alpha=0.25)
#         # axes.plot(t, y-model(s_component[3:], t), '.',
#         #           color=component_colors[i], alpha=0.25)
#         axes.plot(t_fine, m, color=component_colors[i], alpha=0.25)
#
# fig.show()
# # Plot per-component models
# p_mean = np.mean(samples, axis=0)
# p_common = p_mean[:3]
# p_others = p_mean[3:]
# j = 0
# for i, data in enumerate(data_set):
#     t, y = data[:, 0], data[:, 1]
#     p_component = p_others[j: j + models_dims[i]]
#     j += models_dims[i]
#     # axes.plot(t_fine, model(p_component, t_fine), color=component_colors[i])
#     axes.plot(t, y - model(p_component, t), '.', color=component_colors[i])

# # Plot original observed points
# for i, data in enumerate(data_set):
#     t, y = data[:, 0], data[:, 1]
#     axes.plot(t, y, '.k')
#
# fig.show()
#
# # Plot wanted result
# fig, axes = plt.subplots(1, 1)
# axes.set_ylabel("Distance, mas")
# axes.set_xlabel("Time, years")
# for i, s in enumerate(samples[np.random.randint(len(samples), size=6)]):
#     print("Plotting sample {}".format(i))
#     s_common = s[:3]
#     s_others = s[3:]
#     j = 0
#     for i, data in enumerate(data_set):
#         t, y = data[:, 0], data[:, 1]
#         s_component = list(s_common) + list(s_others[j: j+models_dims[i]])
#         j += models_dims[i]
#         gp = george.GP(np.exp(s_component[0]) *
#                        george.kernels.ExpSquaredKernel(np.exp(s_component[1])))
#         # gp = george.GP(np.exp(s[0]) * george.kernels.RationalQuadraticKernel(np.exp(s[1]), np.exp(s[2])))
#         gp.compute(t, 0.001)
#         m = gp.sample_conditional(y - model(s_component[3:], t), t_fine)
#         # axes.plot(t, m, color="#4682b4", alpha=0.25)
#         axes.plot(t_fine, m, color=component_colors[i], alpha=0.25)