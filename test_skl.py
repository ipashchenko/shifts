import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (loocv_poly, cv_ridge_sklearn, cv_svr_sklearn,
                   cv_nusvr_sklearn, cv_kridge_sklearn, cv_lasso_sklearn,
                   cv_elastic_sklearn)


source = '1226+023'
# source = '1334-127'
# source = '0823+033'
n_min_epochs = 10
k=None
data_dir = '/home/ilya/github/shifts/data/errors'
files = glob.glob(os.path.join(data_dir, "{}*kinem.txt".format(source)))
column_names = ["time", "position", 'error']
data_set = list()
comp_ids = list()
for fn in files:
    _, fn_ = os.path.split(fn)
    comp_id = fn_.split('.')[2].split('_')[0][4:]
    df = pd.read_table(fn, delim_whitespace=True, names=column_names,
                       engine='python', usecols=[0, 1, 2], skiprows=2)
    if len(df) >= n_min_epochs:
        data_set.append(df)
        comp_ids.append(comp_id)

data_set = sorted(data_set, key=lambda df: len(df), reverse=True)


number_of_plots = 11
for i, df in enumerate(data_set):
    fig, axes = plt.subplots(1, 1)
    fig.set_size_inches(18.5, 10.5, forward=True)
    colormap = plt.cm.nipy_spectral  # I suggest to use nipy_spectral, Set1,Paired
    axes.set_color_cycle([colormap(j) for j in
                          np.linspace(0, 1, number_of_plots)])
    tt = np.linspace(df['time'].values[0], df['time'].values[-1], 300)

    _, trend_tt_ridge = cv_ridge_sklearn(df['time'].values,
                                         df['position'].values,
                                         yerr=df['error'].values,
                                         max_p=7, t_plot=tt, k=k)
    _, trend_tt_ridge_woerr = cv_ridge_sklearn(df['time'].values,
                                         df['position'].values,
                                         max_p=7, t_plot=tt, k=k)
    # Don't use lasso & elnet because can't treat errors with them
    _, trend_tt_lasso = cv_lasso_sklearn(df['time'].values,
                                         df['position'].values,
                                         max_p=7, t_plot=tt, k=k)
    _, trend_tt_elnet = cv_elastic_sklearn(df['time'].values,
                                           df['position'].values,
                                           max_p=7, t_plot=tt, k=k)

    # _, trend_tt_svr_poly = cv_svr_sklearn(df['time'].values, df['position'].values,
    #                                           kernel='poly', max_p=3, t_plot=tt, k=5)

    # This 3 using rbf kernel and different epsilon
    _, trend_tt_svr_eps1 = cv_svr_sklearn(df['time'].values,
                                          df['position'].values,
                                          yerr=df['error'].values,
                                          kernel='rbf', t_plot=tt, k=k)
    _, trend_tt_svr_eps2 = cv_svr_sklearn(df['time'].values,
                                          df['position'].values,
                                          yerr=df['error'].values,
                                          kernel='rbf', t_plot=tt, k=k,
                                          epsilon=0.2)
    _, trend_tt_svr_eps3 = cv_svr_sklearn(df['time'].values,
                                          df['position'].values,
                                          yerr=df['error'].values,
                                          kernel='rbf', t_plot=tt, k=k,
                                          epsilon=0.3)
    # This 3 using poly kernel and different epsilon
    _, trend_tt_svr_poly_eps1 = cv_svr_sklearn(df['time'].values,
                                               df['position'].values,
                                               yerr=df['error'].values,
                                               kernel='poly', t_plot=tt,
                                               max_p=3, k=k)
    _, trend_tt_svr_poly_eps2 = cv_svr_sklearn(df['time'].values,
                                               df['position'].values,
                                               yerr=df['error'].values,
                                               kernel='poly', t_plot=tt,
                                               max_p=3, k=k, epsilon=0.2)
    _, trend_tt_svr_poly_eps3 = cv_svr_sklearn(df['time'].values,
                                               df['position'].values,
                                               yerr=df['error'].values,
                                               kernel='poly', t_plot=tt,
                                               max_p=3, k=k, epsilon=0.3)

    # Kernel ridge with RBF and RQ kernel
    _, trend_tt_kr_rbf = cv_kridge_sklearn(df['time'].values,
                                           df['position'].values,
                                           yerr=df['error'].values,
                                           kernel='rbf', t_plot=tt, k=k)
    _, trend_tt_kr_rbf_woerr = cv_kridge_sklearn(df['time'].values,
                                           df['position'].values,
                                           kernel='rbf', t_plot=tt, k=k)
    # _, trend_tt_kr_rq = cv_kridge_sklearn(df['time'].values, df['position'].values,
    #                                        kernel='rq', t_plot=tt, k=k)

    axes.plot(df['time'].values, df['position'].values, '.k')
    axes.plot(tt, trend_tt_ridge, label='ridge')
    axes.plot(tt, trend_tt_ridge_woerr, label='ridge wo errors')
    axes.plot(tt, trend_tt_lasso, label='lasso')
    axes.plot(tt, trend_tt_elnet, label='elastic net')
    axes.plot(tt, trend_tt_svr_poly_eps1, label='SVR poly p_max=3 eps=0.1')
    axes.plot(tt, trend_tt_svr_poly_eps2, label='SVR poly p_max=3 eps=0.2')
    axes.plot(tt, trend_tt_svr_poly_eps3, label='SVR poly p_max=3 eps=0.3')
    axes.plot(tt, trend_tt_svr_eps1, label='SVR rbf eps=0.1')
    axes.plot(tt, trend_tt_svr_eps1, label='SVR rbf eps=0.2')
    axes.plot(tt, trend_tt_svr_eps1, label='SVR rbf eps=0.3')
    axes.plot(tt, trend_tt_kr_rbf, label='kernel ridge rbf')
    axes.plot(tt, trend_tt_kr_rbf_woerr, label='kernel ridge rbf wo errors')
    # axes.plot(tt, trend_tt_kr_rq, label='kernel ridge rq')
    # axes.plot(tt, trend_tt_, label='SVR 0.2')
    # axes.plot(tt, trend_tt__, label='SVR 0.3')
    # plt.plot(tt, trend_ttkr, label='KR')
    # plt.plot(tt, trend_ttr, label='R')
    axes.legend(loc='upper left')
    fig.tight_layout()
    fig.show()
    fig.savefig('test_{}_{}_fitting.png'.format(source, i+1), dpi=500)
