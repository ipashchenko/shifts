import os
import numpy as np
from scipy.linalg import cholesky, cho_solve
import scipy.optimize as op
import george
import matplotlib.pyplot as plt
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
data9 = np.loadtxt(os.path.join(data_dir, 'mojave_kinem_comp9.txt'),
                   usecols=[5, 7])
y = data9[:, 1]
x = data9[:, 0]
x_min = np.min(x)
x -= x_min
x_max = np.max(x)
x_test = np.linspace(0, x_max, 1000)

p = np.polyfit(y, x, 2)
y -= np.polyval(p, x)

k1 = 0.2**2 * george.kernels.ExpSquaredKernel(0.5**2)
k2 = 0.5**2 * george.kernels.ExpSquaredKernel(10**2)
k3 = george.kernels.WhiteKernel(0.1**2)
k = k1 + k2 + k3
gp = george.GP(k)


# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.kernel.vector = p
    ll = gp.lnlikelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


# And the gradient of the objective function.
def grad_nll(p):
    gp.kernel.vector = p
    return -gp.grad_lnlikelihood(y, quiet=True)


gp.compute(x)
print(gp.lnlikelihood(y))
p0 = gp.kernel.vector
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
gp.kernel.vector = results.x
print(gp.lnlikelihood(y))

mu, var = gp.predict(y, x_test)
std = np.sqrt(np.diag(var))

mu1, cov1 = partial_predict(k1, k, x, y, x_test)
std1 = np.sqrt(np.diag(cov1))
mu2, cov2 = partial_predict(k2, k, x, y, x_test)
std2 = np.sqrt(np.diag(cov2))
mu3, cov3 = partial_predict(k3, k, x, y, x_test)
std3 = np.sqrt(np.diag(cov3))

# plt.plot(x, y, ".k")
flux = np.loadtxt(os.path.join(data_dir, 'mojave_flux_core.txt'),
                  usecols=[4, 5])
flux[:, 0] -= x_min
flux[:, 1] = (flux[:, 1] - np.median(flux[:, 1]))/(2*mad_std(flux[:, 1]))
fig, axes1 = plt.subplots(1, 1)
axes1.plot(flux[:, 0], flux[:, 1], '.r', label="Core flux")
axes1.plot(flux[:, 0], flux[:, 1], 'r')
# plt.fill_between(x_test, mu+std, mu-std, color="g", alpha=0.25)
# mu1 *= 3
axes2 = axes1.twinx()
axes2.fill_between(x_test, mu1+std1, mu1-std1, color="g", alpha=0.25,
                   label="Short-scale")
axes2.fill_between(x_test, mu2+std2, mu2-std2, color="y", alpha=0.25,
                   label="Long-scale")
axes2.fill_between(x_test, mu3+std3, mu3-std3, color="b", alpha=0.25,
                   label="White Noise")
axes1.set_xlabel("Time from start, years")
axes1.set_ylabel("Core flux, Jy", color='r')
axes1.tick_params('y', colors='r')
axes2.set_ylabel("Shift of core, mas")
# axes2.set_ylim([-0.5, 0.5])
# axes1.set_xlim([0, 12])
axes1.legend(loc="best")
axes2.legend(loc="best")
fig.show()