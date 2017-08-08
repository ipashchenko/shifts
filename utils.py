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
    for p_cur in range(1, max_p+1):
        print("Checking p = {}".format(p_cur))
        cv_score = list()
        for i in range(len(y)):
            print("i = {}".format(i))
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
            y_test = np.array(y[i])
            t_test = np.array(t[i])
            print("y_test = {}, t_test = {}".format(y_test, t_test))
            mm_scaler_t = preprocessing.MinMaxScaler()
            mm_scaler_y = preprocessing.MinMaxScaler()
            transformed_t_ = mm_scaler_t.fit_transform(t_.reshape(-1, 1))[:, 0]
            transformed_t_ -= 0.5
            transformed_y_ = mm_scaler_y.fit_transform(y_.reshape(-1, 1))[:, 0]
            p = np.polyfit(transformed_t_, transformed_y_, p_cur, w=w)
            transformed_t_test = mm_scaler_t.transform(t_test.reshape(1, -1))[:, 0]
            transformed_t_test -= 0.5
            transformed_y_pred = np.polyval(p, transformed_t_test).reshape(1, -1)
            y_pred = mm_scaler_y.inverse_transform(transformed_y_pred)[0, 0]
            cv_score.append((y_test - y_pred)/yerr_test)
        cv_score = np.array(cv_score)
        cv_score = np.mean((cv_score**2))
        cv_scores.append(cv_score)

    return cv_scores


# FIXME: Use max_p high but find the lowest power which has CV within sigma of
# the best CV
def cv_ridge_sklearn(t, y, yerr=None, max_p=5, k=None, t_plot=None):
    from sklearn.linear_model import Ridge
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import KFold, GridSearchCV

    estimator = Ridge()

    if t_plot is None:
        x_plot = np.linspace(t.min(), t.max(), 500)
    else:
        x_plot = t_plot

    if k is None:
        k = len(t) // 2
        k = min(k, 10)
        k = max(k, 4)
    cv = KFold(k, random_state=42, shuffle=False)
    pipeline = Pipeline([('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
                         ('poly', preprocessing.PolynomialFeatures()),
                         ('est', estimator)])
    parameters = {'poly__degree': np.arange(1, max_p+1),
                  'est__alpha': np.logspace(-12, 2, 50)}
    # Choose weights that their sum is len(data) but value ~ 1/error
    if yerr is not None:
        weights = 1/yerr
        weights = weights/(np.sum(weights)/len(weights))
        fit_params = {'est__sample_weight': weights}
    else:
        fit_params = {}

    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1,
                               verbose=1, scoring="neg_mean_squared_error",
                               refit=True, fit_params=fit_params)
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    cv_result = grid_search.fit(np.atleast_2d(t).T, y)
    # id_best = np.where(cv_result['rank_test_score'] == 1)[0][0]
    # best_p = np.arange(1, max_p+1)[id_best]
    # best_p_upper = cv_result['mean_test_score'][id_best] - cv_result['std_test_score'][id_best]
    # id_best = np.where(cv_result['mean_test_score'] - best_p_upper < 0)[0][-1]
    # best_p = np.arange(1, max_p + 1)[id_best]
    #
    # # print("done in %0.3fs" % (time() - t0))
    # print("Best p = {}".format(best_p))

    # print("Best score: %0.7f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # FIXME: With refit=True in GS i can just make prediction with GS but
    # i need simplest parameters (``degree``) with close CV-score actually
    best_parameters = find_simply_best(cv_result)
    print("Simplest and close to best is: ")
    print(best_parameters)
    pipeline.set_params(**best_parameters)
    pipeline.fit(np.atleast_2d(t).T, y)
    y_plot = pipeline.predict(x_plot[:, np.newaxis])
    y_pred = pipeline.predict(np.atleast_2d(t).T)

    return y_pred, y_plot


def find_simply_best(cv_result):
    """
    Among results of CV find simplest parameters (i.e. with low ``poly_degree``)
    that close to the best one within it's std.

    :param cv_result:
        Result of ``sklearn.model_selection._search.GridSearchCV`` fit.
    :return:
        Dictionary with best parameters.
    """
    best_params = cv_result.best_params_
    ix_best = np.where(cv_result.cv_results_['rank_test_score'] == 1)[0][0]

    best_p = cv_result.cv_results_['param_poly__degree'][ix_best]

    best_score = cv_result.cv_results_['mean_test_score'][ix_best]
    best_score_std = cv_result.cv_results_['std_test_score'][ix_best]
    crit_value = best_score - best_score_std

    ix_candidates = np.where(cv_result.cv_results_['mean_test_score'] >
                             crit_value)[0]
    degree_candidates = cv_result.cv_results_['param_poly__degree'][ix_candidates]
    ix_smaller_degree = degree_candidates < best_p
    if np.any(ix_smaller_degree):
        simplest_candidates = ix_candidates[ix_smaller_degree]
        ix_simplest = simplest_candidates[np.argmax(cv_result.cv_results_['mean_test_score'][simplest_candidates])]
        best_params = cv_result.cv_results_['params'][ix_simplest]

    return best_params


def cv_lasso_sklearn(t, y, yerr=None, max_p=5, do_weight_loocv=False, k=None,
                     t_plot=None):
    from sklearn.linear_model import Lasso
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV
    from pprint import pprint
    from time import time

    estimator = Lasso()
    lw = 2
    if t_plot is None:
        x_plot = np.linspace(t.min(), t.max(), 500)
    else:
        x_plot = t_plot
    # cv = LeaveOneOut()
    if k is None:
        k = len(t) // 2
        k = min(k, 10)
        k = max(k, 5)
    cv = KFold(k, random_state=42, shuffle=False)
    pipeline = Pipeline([('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
                         ('poly', preprocessing.PolynomialFeatures()),
                         ('est', estimator)])
    parameters = {'poly__degree': np.arange(1, max_p+1),
                  'est__alpha': np.logspace(-12, 2, 50)}
    if yerr is not None:
        fit_params = {'est__sample_weight': 1/yerr**2}
    else:
        fit_params = {}
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1,
                               verbose=1, scoring="neg_mean_squared_error",
                               refit=True, fit_params=fit_params)
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    cv_result = grid_search.fit(np.atleast_2d(t).T, y).cv_results_
    # id_best = np.where(cv_result['rank_test_score'] == 1)[0][0]
    # best_p = np.arange(1, max_p+1)[id_best]
    # best_p_upper = cv_result['mean_test_score'][id_best] - cv_result['std_test_score'][id_best]
    # id_best = np.where(cv_result['mean_test_score'] - best_p_upper < 0)[0][-1]
    # best_p = np.arange(1, max_p + 1)[id_best]
    #
    # # print("done in %0.3fs" % (time() - t0))
    # print("Best p = {}".format(best_p))

    # print("Best score: %0.7f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # FIXME: With refit=True in GS i can just make prediction with GS
    pipeline.set_params(**best_parameters)
    pipeline.fit(np.atleast_2d(t).T, y)
    y_plot = pipeline.predict(x_plot[:, np.newaxis])
    y_pred = pipeline.predict(np.atleast_2d(t).T)
    # print "Setted params"
    # print pipeline.get_params()
    # plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
    #          linewidth=lw, label='%s' % (name))
    # plt.plot(x_plot, y_plot, linewidth=lw)
    # plt.plot(t, y, '.', linewidth=lw)
    # legend = plt.legend(loc='upper left', frameon=False)
    return y_pred, y_plot


def cv_elastic_sklearn(t, y, yerr=None, max_p=5, do_weight_loocv=False, k=None,
                       t_plot=None):
    from sklearn.linear_model import ElasticNet
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    from sklearn.model_selection import LeaveOneOut, KFold, GridSearchCV
    from pprint import pprint
    from time import time

    estimator = ElasticNet()
    lw = 2
    if t_plot is None:
        x_plot = np.linspace(t.min(), t.max(), 500)
    else:
        x_plot = t_plot
    # cv = LeaveOneOut()
    if k is None:
        k = len(t) // 2
        k = min(k, 10)
        k = max(k, 5)
    cv = KFold(k, random_state=42, shuffle=False)
    pipeline = Pipeline([('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
                         ('poly', preprocessing.PolynomialFeatures()),
                         ('est', estimator)])
    parameters = {'poly__degree': np.arange(1, max_p+1),
                  'est__alpha': np.logspace(-3, 2, 20),
                  'est__l1_ratio': np.linspace(0.01, 0.99, 15)}
    if yerr is not None:
        fit_params = {'est__sample_weight': 1/yerr**2}
    else:
        fit_params = {}
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1,
                               verbose=1, scoring="neg_mean_squared_error",
                               refit=True, fit_params=fit_params)
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    cv_result = grid_search.fit(np.atleast_2d(t).T, y).cv_results_
    # id_best = np.where(cv_result['rank_test_score'] == 1)[0][0]
    # best_p = np.arange(1, max_p+1)[id_best]
    # best_p_upper = cv_result['mean_test_score'][id_best] - cv_result['std_test_score'][id_best]
    # id_best = np.where(cv_result['mean_test_score'] - best_p_upper < 0)[0][-1]
    # best_p = np.arange(1, max_p + 1)[id_best]
    #
    # # print("done in %0.3fs" % (time() - t0))
    # print("Best p = {}".format(best_p))

    # print("Best score: %0.7f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # FIXME: With refit=True in GS i can just make prediction with GS
    pipeline.set_params(**best_parameters)
    pipeline.fit(np.atleast_2d(t).T, y)
    y_plot = pipeline.predict(x_plot[:, np.newaxis])
    y_pred = pipeline.predict(np.atleast_2d(t).T)
    # print "Setted params"
    # print pipeline.get_params()
    # plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
    #          linewidth=lw, label='%s' % (name))
    # plt.plot(x_plot, y_plot, linewidth=lw)
    # plt.plot(t, y, '.', linewidth=lw)
    # legend = plt.legend(loc='upper left', frameon=False)
    return y_pred, y_plot


def cv_svr_sklearn(t, y, yerr=None, kernel='poly', max_p=5,
                   do_weight_loocv=False, k=None, t_plot=None, epsilon=0.1):
    from sklearn.svm import SVR
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold, GridSearchCV
    from time import time

    estimator = SVR(kernel=kernel, tol=0.001, cache_size=1000,
                    max_iter=100000, epsilon=epsilon, verbose=False)
    lw = 2
    if t_plot is None:
        x_plot = np.linspace(t.min(), t.max(), 500)
    else:
        x_plot = t_plot
    # cv = LeaveOneOut()
    if k is None:
        k = len(t) // 2
        k = min(k, 10)
        k = max(k, 5)
    cv = KFold(k, random_state=42, shuffle=False)
    pipeline = Pipeline([('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
                         ('est', estimator)])
    parameters = {'est__C': np.logspace(-3, 3, 20),
                  'est__gamma': np.logspace(-3, 3, 20)}
    if kernel == 'poly':
        parameters.update({'est__degree': np.arange(1, max_p+1)})
    if yerr is not None:
        fit_params = {'est__sample_weight': 1 / yerr ** 2}
    else:
        fit_params = {}
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1,
                               scoring="neg_mean_squared_error",
                               refit=True, fit_params=fit_params)
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    cv_result = grid_search.fit(np.atleast_2d(t).T, y).cv_results_
    # id_best = np.where(cv_result['rank_test_score'] == 1)[0][0]
    # best_p = np.arange(1, max_p+1)[id_best]
    # best_p_upper = cv_result['mean_test_score'][id_best] - cv_result['std_test_score'][id_best]
    # id_best = np.where(cv_result['mean_test_score'] - best_p_upper < 0)[0][-1]
    # best_p = np.arange(1, max_p + 1)[id_best]
    #
    # # print("done in %0.3fs" % (time() - t0))
    # print("Best p = {}".format(best_p))

    # print("Best score: %0.7f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # FIXME: With refit=True in GS i can just make prediction with GS
    pipeline.set_params(**best_parameters)
    pipeline.fit(np.atleast_2d(t).T, y)
    y_plot = pipeline.predict(x_plot[:, np.newaxis])
    y_pred = pipeline.predict(np.atleast_2d(t).T)
    # print "Setted params"
    # print pipeline.get_params()
    # plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
    #          linewidth=lw, label='%s' % (name))
    # plt.plot(x_plot, y_plot, linewidth=lw)
    # plt.plot(t, y, '.', linewidth=lw)
    # legend = plt.legend(loc='upper left', frameon=False)
    return y_pred, y_plot


def cv_nusvr_sklearn(t, y, yerr=None, kernel='poly', max_p=5,
                     do_weight_loocv=False, k=None, t_plot=None):
    from sklearn.svm import NuSVR
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold, GridSearchCV
    from time import time

    estimator = NuSVR(kernel=kernel, tol=0.001, cache_size=1000,
                      max_iter=100000)
    lw = 2
    if t_plot is None:
        x_plot = np.linspace(t.min(), t.max(), 500)
    else:
        x_plot = t_plot
    # cv = LeaveOneOut()
    if k is None:
        k = len(t) // 2
        k = min(k, 10)
        k = max(k, 5)
    cv = KFold(k, random_state=42, shuffle=False)
    pipeline = Pipeline([('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
                         ('est', estimator)])
    parameters = {'est__C': np.logspace(-3, 3, 10),
                  'est__gamma': np.logspace(-3, 3, 10),
                  'est__nu': np.linspace(0.1, 0.9, 10)}
    if kernel == 'poly':
        parameters.update({'est__degree': np.arange(1, max_p+1)})
    if yerr is not None:
        fit_params = {'est__sample_weight': 1/yerr**2}
    else:
        fit_params = {}
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1,
                               scoring="neg_mean_squared_error",
                               refit=True, fit_params=fit_params)
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    cv_result = grid_search.fit(np.atleast_2d(t).T, y).cv_results_
    # id_best = np.where(cv_result['rank_test_score'] == 1)[0][0]
    # best_p = np.arange(1, max_p+1)[id_best]
    # best_p_upper = cv_result['mean_test_score'][id_best] - cv_result['std_test_score'][id_best]
    # id_best = np.where(cv_result['mean_test_score'] - best_p_upper < 0)[0][-1]
    # best_p = np.arange(1, max_p + 1)[id_best]
    #
    # # print("done in %0.3fs" % (time() - t0))
    # print("Best p = {}".format(best_p))

    # print("Best score: %0.7f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # FIXME: With refit=True in GS i can just make prediction with GS
    pipeline.set_params(**best_parameters)
    pipeline.fit(np.atleast_2d(t).T, y)
    y_plot = pipeline.predict(x_plot[:, np.newaxis])
    y_pred = pipeline.predict(np.atleast_2d(t).T)
    # print "Setted params"
    # print pipeline.get_params()
    # plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
    #          linewidth=lw, label='%s' % (name))
    # plt.plot(x_plot, y_plot, linewidth=lw)
    # plt.plot(t, y, '.', linewidth=lw)
    # legend = plt.legend(loc='upper left', frameon=False)
    return y_pred, y_plot


def cv_kridge_sklearn(t, y, yerr=None, kernel='rbf', do_weight_loocv=False,
                      k=None, t_plot=None):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn import preprocessing
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.gaussian_process.kernels import RBF, RationalQuadratic

    from time import time

    estimator = KernelRidge()
    lw = 2
    if t_plot is None:
        x_plot = np.linspace(t.min(), t.max(), 500)
    else:
        x_plot = t_plot
    # cv = LeaveOneOut()
    if k is None:
        k = len(t) // 2
        k = min(k, 10)
        k = max(k, 5)
    cv = KFold(k, random_state=42, shuffle=False)
    pipeline = Pipeline([('scale', preprocessing.MinMaxScaler(feature_range=(-1, 1))),
                         ('est', estimator)])
    parameters = {'est__alpha': np.logspace(-5, 2, 10),
                  'est__gamma': np.logspace(-9, 2, 10)}
    if kernel == 'rbf':
        parameters.update({'est__kernel':
                               [RBF(l) for l in np.logspace(-1, 1, 20)]})
    if kernel == 'rq':
        parameters.update({'est__kernel':
                               [RationalQuadratic(l, a)
                                for l in np.logspace(-1, 1, 20)
                                for a in np.logspace(-3, 2, 20)]})
    if yerr is not None:
        fit_params = {'est__sample_weight': 1/yerr**2}
    else:
        fit_params = {}
    grid_search = GridSearchCV(pipeline, parameters, cv=cv, n_jobs=-1,
                               verbose=1, scoring="neg_mean_squared_error",
                               refit=True, fit_params=fit_params)
    # print("Performing grid search...")
    # print("pipeline:", [name for name, _ in pipeline.steps])
    # print("parameters:")
    # pprint(parameters)
    t0 = time()
    cv_result = grid_search.fit(np.atleast_2d(t).T, y).cv_results_
    # id_best = np.where(cv_result['rank_test_score'] == 1)[0][0]
    # best_p = np.arange(1, max_p+1)[id_best]
    # best_p_upper = cv_result['mean_test_score'][id_best] - cv_result['std_test_score'][id_best]
    # id_best = np.where(cv_result['mean_test_score'] - best_p_upper < 0)[0][-1]
    # best_p = np.arange(1, max_p + 1)[id_best]
    #
    # # print("done in %0.3fs" % (time() - t0))
    # print("Best p = {}".format(best_p))

    # print("Best score: %0.7f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # FIXME: With refit=True in GS i can just make prediction with GS
    pipeline.set_params(**best_parameters)
    pipeline.fit(np.atleast_2d(t).T, y)
    y_plot = pipeline.predict(x_plot[:, np.newaxis])
    y_pred = pipeline.predict(np.atleast_2d(t).T)
    # print "Setted params"
    # print pipeline.get_params()
    # plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
    #          linewidth=lw, label='%s' % (name))
    # plt.plot(x_plot, y_plot, linewidth=lw)
    # plt.plot(t, y, '.', linewidth=lw)
    # legend = plt.legend(loc='upper left', frameon=False)
    return y_pred, y_plot


def choose_simply_best(cv_result):
    """
    Among results of the ``GridSearchCV`` choose the simplest one (measured by
    parameter ``degree``) that is close to the best within one sigma.

    :param cv_result:
        Result of `GridSearchCV`` with ``est__degree`` being one of the
        optimized parameters.
    :return:

    """
    pass