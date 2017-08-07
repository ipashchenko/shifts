import os
import numpy as np
from scipy.linalg import cholesky, cho_solve
import scipy.optimize as op
import george
import matplotlib.pyplot as plt
from functools import partial
from astropy.stats import mad_std


def partial_predict(K_i, K, x, y, x_star):
    """
    Compute conditional distribution of GP conditioned on its sum with another
    GP
    .
    :param K_i:
        Instance of ``george.Kernel`` class. The kernel we are interested in.
    :param K:
        Instance of ``george.Kernel`` class. Composite kernel that is sum of
        several kernels (including ``K_i``).
    :param x:
        Observations conditioned on.
    :param y:
        Observations conditioned on.
    :param x_star:
        Where to compute conditional distribution.
    """
    x = np.atleast_2d(x).T
    x_star = np.atleast_2d(x_star).T

    # Precompute quantities required for predictions which are independent
    # of actual query points
    K_ = K.value(x, x)
    L_ = cholesky(K_, lower=True)
    alpha_ = cho_solve((L_, True), y)

    K_i_star = K_i.value(x_star, x)
    y_mean = K_i_star.dot(alpha_)
    # y_mean = y + y_mean

    v = cho_solve((L_, True), K_i_star.T)
    y_cov = K_i.value(x_star, x_star) - K_i_star.dot(v)

    return y_mean, y_cov


data_dir = '/home/ilya/github/shifts/data'
n = 12
data9 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp{}.txt'.format(n)),
                   usecols=[5, 7])
y = data9[:, 1]
x = data9[:, 0]
x_min = np.min(x)
x -= x_min
x_max = np.max(x)
x_test = np.linspace(0, x_max, 1000)

p = np.polyfit(x, y, 1)
y -= np.polyval(p, x)

# k1 = 0.2**2 * george.kernels.ExpSquaredKernel(0.25**2)
k1 = 0.2**2 * george.kernels.RationalQuadraticKernel(50, 1**2)
# k2 = 25 * george.kernels.ExpSquaredKernel(20**2)# * george.kernels.ConstantKernel(100.)
k3 = george.kernels.WhiteKernel(0.1**2)
# k4 = george.kernels.DotProductKernel() * george.kernels.ConstantKernel(1.)
k = k1 + k3
gpy = george.GP(k)


# Define the objective function (negative log-likelihood in this case).
def nll(p, y, gp):
    gp.kernel.vector = p
    ll = gp.lnlikelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


# And the gradient of the objective function.
def grad_nll(p, y, gp):
    gp.kernel.vector = p
    return -gp.grad_lnlikelihood(y, quiet=True)


nll_y = partial(nll, y=y, gp=gpy)
grad_nll_y = partial(grad_nll, y=y, gp=gpy)

gpy.compute(x)
print(gpy.lnlikelihood(y))
p0 = gpy.kernel.vector
results = op.minimize(nll_y, p0, jac=grad_nll_y, method="L-BFGS-B")
gpy.kernel.vector = results.x
print(gpy.lnlikelihood(y))

# Now fit flux
flux = np.loadtxt(os.path.join(data_dir, 'mojave_flux_core.txt'),
                  usecols=[4, 5])
flux[:, 0] -= x_min
tf = flux[:, 0]
tf_test = np.linspace(tf[0], tf[-1], 1000)
kf1 = 2**2 * george.kernels.ExpSquaredKernel(1**2)
# kf1 = 1**2 * george.kernels.RationalQuadraticKernel(0.1, 5**2)
# kf2 = 1**2 * george.kernels.ExpSquaredKernel(20**2)# * george.kernels.ConstantKernel(100.)
kf3 = george.kernels.WhiteKernel(0.15**2)
# k4 = george.kernels.DotProductKernel() * george.kernels.ConstantKernel(1.)
kf = kf1 + kf3
gpf = george.GP(kf)
gpf.compute(tf)

yf = flux[:, 1] - np.median(flux[:, 1])

p0 = gpf.kernel.vector

nll_f = partial(nll, y=yf, gp=gpf)
grad_nll_f = partial(grad_nll, y=yf, gp=gpf)

results = op.minimize(nll_f, p0, jac=grad_nll_f, method="L-BFGS-B")
gpf.kernel.vector = results.x
print(gpf.lnlikelihood(yf))

muf, varf = gpf.predict(yf, tf_test)
stdf = np.sqrt(np.diag(varf))

mu, var = gpy.predict(y, x_test)
std = np.sqrt(np.diag(var))

mu1, cov1 = partial_predict(k1, k, x, y, x_test)
std1 = np.sqrt(np.diag(cov1))
# mu2, cov2 = partial_predict(k2, k, x, y, x_test)
# std2 = np.sqrt(np.diag(cov2))
mu3, cov3 = partial_predict(k3, k, x, y, x_test)
std3 = np.sqrt(np.diag(cov3))
# mu4, cov4 = partial_predict(k4, k, x, y, x_test)
# std4 = np.sqrt(np.diag(cov4))

# plt.plot(x, y, ".k")

# flux[:, 1] = (flux[:, 1] - np.median(flux[:, 1]))/(2*mad_std(flux[:, 1]))
fig, axes1 = plt.subplots(1, 1)
axes1.plot(flux[:, 0]+x_min, flux[:, 1], '.r', label="Core flux")
# axes1.plot(flux[:, 0], flux[:, 1], 'r')
axes1.plot(tf_test+x_min, muf+np.median(flux[:, 1]), color="r")
axes1.fill_between(tf_test+x_min, muf+stdf+np.median(flux[:, 1]),
                   muf-stdf+np.median(flux[:, 1]), color="r", alpha=0.25)
# plt.fill_between(x_test, mu+std, mu-std, color="g", alpha=0.25)
# mu1 *= 3
axes2 = axes1.twinx()
axes2.fill_between(x_test+x_min, mu1+std1, mu1-std1, color="g", alpha=0.25)
axes2.plot(x_test+x_min, mu1, color="g", label="Shift")
# axes2.fill_between(x_test, mu2+std2, mu2-std2, color="b", alpha=0.25,
#                    label="2")
# axes2.fill_between(x_test, mu4+std4, mu4-std4, color="r", alpha=0.25,
#                    label="4")
# axes2.fill_between(x_test, mu3+std3, mu3-std3, color="b", alpha=0.25,
#                    label="White Noise")
axes1.set_xlabel("Time, year")
axes1.set_ylabel("Core flux, Jy", color='r')
axes1.tick_params('y', colors='r')
axes2.set_ylabel("Shift of core, mas", color='g')
axes2.tick_params('y', colors='g')
# axes2.set_ylim([-0.75, 0.75])
# axes1.set_xlim([0, 12])
axes1.legend(loc="lower left")
axes2.legend(loc="lower right")
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(data_dir, 'comp{}_decomposition_exp.png'.format(n)),
            dpi=500)

fig, axes = plt.subplots(1, 1)
axes.fill_between(x_test, mu+std, mu-std, color="g", label="all", alpha=0.25)
axes.fill_between(x_test, mu1+std1, mu1-std1, color="r", alpha=0.25,
                   label="1")
# axes.fill_between(x_test, mu2+std2, mu2-std2, color="b", alpha=0.25,
#                    label="2")
# axes.fill_between(x_test, mu4+std4, mu4-std4, color="r", alpha=0.25,
#                    label="4")
axes.plot(x, y, '.k')
axes.plot(x, y, 'k')
axes.legend(loc="best")
fig.show()