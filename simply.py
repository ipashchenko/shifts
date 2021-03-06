import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import uniform
from functools import partial
from sklearn import preprocessing
import matplotlib.pyplot as plt
# from utils import partial_predict, optimize_gp
from utils import loocv_poly


max_p = 3
# Find all sources
data_dir = '/home/ilya/github/shifts/data/all'
files = glob.glob(os.path.join(data_dir, "*kinem.txt"))
sources = list()
for fn in files:
    _, fn_ = os.path.split(fn)
    sources.append(fn_.split('.')[0])
sources = list(set(sources))
# sources = ['1226+023']

for source in sources:
    n_min_epochs = 7
    data_dir = '/home/ilya/github/shifts/data/errors'
    files = glob.glob(os.path.join(data_dir, "{}*kinem.txt".format(source)))
    column_names = ["time", "position"]
    data_set = list()
    comp_ids = list()
    for fn in files:
        _, fn_ = os.path.split(fn)
        comp_id = fn_.split('.')[2].split('_')[0][4:]
        df = pd.read_table(fn, delim_whitespace=True, names=column_names,
                           engine='python', usecols=[0, 1])
        if len(df) >= n_min_epochs:
            data_set.append(df)
            comp_ids.append(comp_id)

    fig, axes = plt.subplots(1, 1)
    for comp_id, data in zip(comp_ids, data_set):
        axes.plot(data['time'], data['position'], '.', label=comp_id)
    axes.set_xlabel("Time, years")
    axes.set_ylabel("Component distance, mas")
    axes.legend(loc="best")
    axes.set_title(source)
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(data_dir, '{}_raw.png'.format(source)), dpi=300)
    plt.close(fig)

    data_set = sorted(data_set, key=lambda df: len(df), reverse=True)

    # This for plotting average deviations
    # fig, axes = plt.subplots(1, 1)
    # This for plotting data with best fitted polynomials
    fig_, axes_ = plt.subplots(1, 1)
    powers = dict()
    for i, df in enumerate(data_set):
        mm_scaler_t = preprocessing.MinMaxScaler()
        mm_scaler_y = preprocessing.MinMaxScaler()
        # TODO: Here optionally choose the order of polynom
        cv_scores = loocv_poly(df['position'].values, df['time'].values,
                               max_p=max_p)
        p_power = np.argmin(cv_scores)
        powers[comp_ids[i]] = p_power+1
        p = np.polyfit(mm_scaler_t.fit_transform(df['time'])-0.5,
                       mm_scaler_y.fit_transform(df['position']), p_power+1)
                       # w=df['flux'])
        print(p)
        trend = mm_scaler_y.inverse_transform(np.polyval(p, mm_scaler_t.transform(df['time'])-0.5))
        tt = np.linspace(df['time'].values[0], df['time'].values[-1], 300)
        trend_tt = mm_scaler_y.inverse_transform(np.polyval(p, mm_scaler_t.transform(tt)-0.5))
        axes_.plot(tt, trend_tt, '-', label=comp_ids[i])
        axes_.plot(df['time'], df['position'], '.', label="")
        df['position'] -= trend
        data_set[i] = df
        # axes.plot(df['time'], df['position'])
        # axes.plot(df['time'], df['position'], '.')
    axes_.legend(loc='best')
    fig_.show()
    fig_.savefig(os.path.join(data_dir, '{}_raw_fitted.png'.format(source)), dpi=300)
    plt.close(fig_)

    # Average residuals from fit
    data = data_set[0]
    for i, df in enumerate(data_set[1:]):
        data = data.join(df.set_index('time'), on='time', lsuffix='_{}'.format(i+1),
                         rsuffix='_{}'.format(i+2))

    adata = np.array(data)

    # Plot averaged detrended positions vs Core Flux
    fig, axes = plt.subplots(1, 1)
    minmax_scaler_shift = preprocessing.MinMaxScaler()
    mean_deviations = np.nanmean(adata[:, 1:], axis=1)
    axes.plot(adata[:, 0], minmax_scaler_shift.fit_transform(mean_deviations), '.g')
    axes.plot(adata[:, 0], minmax_scaler_shift.transform(mean_deviations), 'g')

    flux = pd.read_table(os.path.join(data_dir,
                                      '{}.u1.core_flux.txt'.format(source)),
                         delim_whitespace=True, names=["time", "flux"],
                         engine='python', usecols=[0, 1])
    # idx = np.where(flux[:, 0] > data['time'][0])
    # axes.plot(flux[:, 0][idx], preprocessing.minmax_scale(flux[:, 1][idx]), 'r')
    # axes.plot(flux[:, 0][idx], preprocessing.minmax_scale(flux[:, 1][idx]), '.r')
    axes2 = axes.twinx()
    minmax_scaler_flux = preprocessing.MinMaxScaler()
    axes2.plot(flux["time"], minmax_scaler_flux.fit_transform(flux["flux"]), 'r')
    axes2.plot(flux["time"], minmax_scaler_flux.transform(flux["flux"]), '.r')
    axes.set_xlabel("Time")
    axes2.set_ylabel("Flux, Jy", color='r')
    axes.tick_params('y', colors='g')
    axes2.tick_params('y', colors='r')

    normed_flux = axes2.yaxis.get_majorticklocs()
    real_flux = minmax_scaler_flux.inverse_transform(normed_flux[:])
    axes2.set_yticklabels(np.around(real_flux, 2))

    normed_shift = axes.yaxis.get_majorticklocs()
    real_shift = minmax_scaler_shift.inverse_transform(normed_shift[:])
    axes.set_yticklabels(np.around(real_shift, 2))

    axes.set_ylabel("Averaged deviations from trend, mas", color='g')
    axes.set_title(source)
    fig.tight_layout()
    fig.show()
    powers_plot = [powers[i] for i in sorted(powers.keys(), key=lambda x: int(x))]
    powers_plot_string = ""
    for s in powers_plot:
        powers_plot_string += str(s)
    fig.savefig(os.path.join(data_dir, '{}_best_{}d.png'.format(source, powers_plot_string)), dpi=300)
    plt.close(fig)


    # Fit shifts
    # x_fine = np.linspace(adata[0, 0], adata[-1, 0], 1000)-adata[0, 0]
    # results = optimize_gp(adata[:, 0]-adata[0, 0], adata[:, 1],
    #                       p0=[0.2, 0.001, 1, 0.1])
    # mu, cov = partial_predict(results['k1'], results['k'], adata[:, 0]-adata[0, 0], adata[:, 1],
    #                           x_fine)
    # std = np.sqrt(np.diag(cov))
    # axes.fill_between(x_fine+adata[0, 0], mu-std, mu+std, color='g', alpha=0.25)
    # fig.show()
    # t_max = np.max([data[-1, 0] for data in data_set])
    # t_fine = np.linspace(0, t_max, 500)
