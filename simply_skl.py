import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import preprocessing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import cv_ridge_sklearn


# FIXME: Use max_p high but find the lowest power which has CV within sigma of
# the best CV
max_p = 3

# This means choose ``k`` adaptively - not large then 10 but not smaller then 4
k = None
# Find all sources
data_dir = '/home/ilya/github/shifts/data/errors'
save_dir = '/home/ilya/github/shifts/data/errors'
files = glob.glob(os.path.join(data_dir, "*kinem.txt"))
sources = list()
for fn in files:
    _, fn_ = os.path.split(fn)
    sources.append(fn_.split('.')[0])
sources = list(set(sources))
# sources = ['0300+470']
sources = ['2209+236']

for source in sources:
    print source
    n_min_epochs = 8
    data_dir = '/home/ilya/github/shifts/data/errors'
    files = glob.glob(os.path.join(data_dir, "{}*kinem.txt".format(source)))
    column_names = ["time", "position", "error"]
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

    # This make figure wo opening window
    fig = Figure()
    axes = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvas(fig)

    for comp_id, data in zip(comp_ids, data_set):
        axes.plot(data['time'], data['position'], '.', label=comp_id)
    axes.set_xlabel("Time, years")
    axes.set_ylabel("Component distance, mas")
    axes.legend(loc="upper left")
    axes.set_title(source)
    fig.tight_layout()
    # fig.show()
    # fig.savefig(os.path.join(save_dir, '{}_raw.png'.format(source)), dpi=300)
    canvas.print_figure(os.path.join(save_dir, '{}_raw.png'.format(source)), dpi=300)
    # plt.close(fig)
    #
    data_set = sorted(data_set, key=lambda df: np.min(df['time'].values))

    # This for plotting average deviations
    # fig, axes = plt.subplots(1, 1)
    # This for plotting data with best fitted polynomials
    fig = Figure()
    axes = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvas(fig)

    for i, df in enumerate(data_set):
        df.loc[df['error'] == 0, 'error'] = 0.01
        mm_scaler_t = preprocessing.MinMaxScaler()
        mm_scaler_y = preprocessing.MinMaxScaler()
        # TODO: Here optionally choose the order of polynom
        tt = np.linspace(df['time'].values[0], df['time'].values[-1], 300)
        trend, trend_tt = cv_ridge_sklearn(df['time'].values,
                                           df['position'].values,
                                           yerr=df['error'].values,
                                           max_p=max_p,
                                           t_plot=tt, k=k)
        axes.plot(tt, trend_tt, '-', label=comp_ids[i])
        axes.plot(df['time'], df['position'], '.', label="")
        df['position'] -= trend
        data_set[i] = df
        # axes.plot(df['time'], df['position'])
        # axes.plot(df['time'], df['position'], '.')
    axes.legend(loc='upper left')
    canvas.print_figure(os.path.join(save_dir, '{}_raw_fitted_ridge.png'.format(source)), dpi=300)

    # Average residuals from fit
    data = data_set[0]
    for i, df in enumerate(data_set[1:]):
        data = data.join(df.set_index('time'), how='outer', on='time',
                         lsuffix='_{}'.format(i+1), rsuffix='_{}'.format(i+2))
    data = data.sort_values('time')
    adata = np.array(data)

    # Plot averaged detrended positions vs Core Flux
    fig = Figure()
    axes = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvas(fig)

    minmax_scaler_shift = preprocessing.MinMaxScaler()
    # FIXME: Use weights in averaging
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
    canvas.print_figure(os.path.join(save_dir, '{}_average.png'.format(source)), dpi=300)


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
