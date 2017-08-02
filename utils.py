import numpy as np
import george
from functools import partial
import scipy.optimize as op
from scipy.linalg import cholesky, cho_solve
from sklearn import preprocessing


# Define the objective function (negative log-likelihood in this case).
def nll(p, y, gp):
    gp.kernel.vector = p
    ll = gp.lnlikelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


# And the gradient of the objective function.
def grad_nll(p, y, gp):
    gp.kernel.vector = p
    return -gp.grad_lnlikelihood(y, quiet=True)


def optimize_gp(y, x, p0=None):
    """
    Optimize GP hyperparameters for given data.
    :param p0: (optional)
        Initial values for the hyperparameters ("Amp", "alpha", "tau", "d").
    :return:
        Dictionary with results.
    """
    if p0 is None:
        p0 = [0.2, 50, 1, 0.1]
    k1 = p0[0] ** 2 * george.kernels.RationalQuadraticKernel(p0[1], p0[2] ** 2)
    k2 = george.kernels.WhiteKernel(p0[3] ** 2)
    k = k1 + k2
    gp = george.GP(k)
    nll_ = partial(nll, y=y, gp=gp)
    grad_nll_ = partial(grad_nll, y=y, gp=gp)

    gp.compute(x)
    print(gp.lnlikelihood(y))
    p0 = gp.kernel.vector
    results = op.minimize(nll_, p0, jac=grad_nll_, method="L-BFGS-B")
    gp.kernel.vector = results.x
    print(gp.lnlikelihood(y))
    return {'gp': gp, 'k': k, 'k1': k1, 'k2': k2, 'gp_lnlik': gp.lnlikelihood(y)}


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


def loocv_poly(y, t, yerr=None, max_p=3, do_weight_loocv=False):
    """
    Leave-One-Out Cross-Validation check of fitting polynomial.

    :param y:
        Itearble of values.
    :param t:
        Iterable of arguments.
    :param yerr: (optional)
        Iterable of errors of values. Used as weights in polynomial fit. If
        ``None`` the don't use weights. (default: ``None``)
    :param max_p:
        Maximum power of fitting polynomial function.
    :param do_weight_loocv: (optional)
        Use weighted MSE as CV-score? (default: ``False``)

    :return:
        Iterable of LOOCV scores that are MSE.
    """
    cv_scores = list()
    for p in range(1, max_p+1):
        cv_score = list()
        for i in range(len(y)):
            y_ = np.delete(y, i)
            t_ = np.delete(t, i)


            if yerr is not None:
                w = 1/np.delete(yerr, i)
                if do_weight_loocv:
                    yerr_test = yerr[i]
                else:
                    yerr_test = 1.
            else:
                w = None
                yerr_test = 1.
            y_test = y[i]
            t_test = t[i]
            mm_scaler_t = preprocessing.MinMaxScaler()
            mm_scaler_y = preprocessing.MinMaxScaler()
            p = np.polyfit(mm_scaler_t.fit_transform(t_) - 0.5,
                           mm_scaler_y.fit_transform(y_), p, w=w)
            y_pred = mm_scaler_y.inverse_transform(np.polyval(p, mm_scaler_t.transform(t_test) - 0.5))
            cv_score.append((y_test - y_pred)/yerr_test)
        cv_score = np.array(cv_score)
        cv_score = np.mean((cv_score**2))
        cv_scores.append(cv_score)

    return cv_scores

