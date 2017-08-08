import os
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import cv_ridge_sklearn, OrderedDefaultDict


max_p = 3

# This will be the Fomalont error where it is specified as ``0``
zero_error_value = 0.01

# This means choose ``k`` adaptively - not large then 10 but not smaller then 4
k = None

data_dir = '/home/ilya/github/shifts/data/errors'
save_dir = '/home/ilya/github/shifts/data/errors'
files = glob.glob(os.path.join(data_dir, "*kinem.txt"))
sources = list()
for fn in files:
    _, fn_ = os.path.split(fn)
    sources.append(fn_.split('.')[0])
sources = list(set(sources))
sources = ['0300+470', '2209+236']
# sources = ['2209+236']
corr_coefs = OrderedDefaultDict()
corr_coefs_w = OrderedDefaultDict()

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
    canvas.print_figure(os.path.join(save_dir, '{}_raw.png'.format(source)), dpi=300)
    data_set = sorted(data_set, key=lambda df: np.min(df['time'].values))

    fig = Figure()
    axes = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvas(fig)

    for i, df in enumerate(data_set):
        df.loc[df['error'] == 0, 'error'] = zero_error_value
        mm_scaler_t = preprocessing.MinMaxScaler()
        mm_scaler_y = preprocessing.MinMaxScaler()
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
    axes.legend(loc='upper left')
    canvas.print_figure(os.path.join(save_dir,
                                     '{}_raw_fitted_ridge.png'.format(source)),
                        dpi=300)

    # Average residuals from fit
    try:
        data = data_set[0]
    except IndexError:
        # Some sources have no components with number of epochs > then minimal
        # required
        continue

    for i, df in enumerate(data_set[1:]):
        data = data.join(df.set_index('time'), how='outer', on='time',
                         lsuffix='_{}'.format(i+1), rsuffix='_{}'.format(i+2))
    data = data.sort_values('time')

    # Plot averaged detrended positions vs Core Flux
    fig = Figure()
    axes = fig.add_subplot(1, 1, 1)
    canvas = FigureCanvas(fig)

    minmax_scaler_shift = preprocessing.MinMaxScaler()
    n_positions = len(data.columns) // 2
    positions = [data[data.columns[2 * i + 1]] for i in range(n_positions)]
    positions = np.vstack(positions).T
    positions = np.ma.masked_array(positions, np.isnan(positions))
    errors = [data[data.columns[2 * i + 2]] for i in range(n_positions)]
    errors = np.vstack(errors).T
    errors = np.ma.masked_array(errors, np.isnan(errors))
    mean_deviations_w = np.ma.average(positions, axis=1, weights=1./errors)
    mean_deviations = np.ma.average(positions, axis=1)
    data['mean_deviations'] = mean_deviations
    data['mean_deviations_w'] = mean_deviations_w

    flux = pd.read_table(os.path.join(data_dir,
                                      '{}.u1.core_flux.txt'.format(source)),
                         delim_whitespace=True, names=["time", "flux"],
                         engine='python', usecols=[0, 1])

    # Find common times in averaged residuals and flux DataFrames
    data_flux = data.join(flux.set_index('time'), how='left', on='time')
    # Find corr. coef between residuals and flux
    corr_pearson = data_flux.corr(method='pearson')
    corr_kendall = data_flux.corr(method='kendall')
    corr_spearman = data_flux.corr(method='spearman')
    corr_coefs[source] = {'pearson': corr_pearson['flux']['mean_deviations'],
                          'kendall': corr_kendall['flux']['mean_deviations'],
                          'spearman': corr_spearman['flux']['mean_deviations']}
    corr_coefs_w[source] = {'pearson': corr_pearson['flux']['mean_deviations_w'],
                            'kendall': corr_kendall['flux']['mean_deviations_w'],
                            'spearman': corr_spearman['flux']['mean_deviations_w']}

    # Labels for plots showing values of corr. coefficients
    label_w = r"w ave $r_{{pear}}$={:.2f}, $r_{{kend}}$={:.2f}," \
              r" $r_{{spear}}$={:.2f}".format(corr_pearson['flux']['mean_deviations_w'],
                                              corr_kendall['flux']['mean_deviations_w'],
                                              corr_spearman['flux']['mean_deviations_w'])
    label = r"$r_{{pear}}$={:.2f}, $r_{{kend}}$={:.2f}," \
            r"$r_{{spear}}$={:.2f}".format(corr_pearson['flux']['mean_deviations'],
                                           corr_kendall['flux']['mean_deviations'],
                                           corr_spearman['flux']['mean_deviations'])
    axes.plot(data['time'],
              minmax_scaler_shift.fit_transform(mean_deviations), '.g')
    axes.plot(data['time'],
              minmax_scaler_shift.transform(mean_deviations), 'g',
              label=label)
    axes.plot(data['time'],
              minmax_scaler_shift.transform(mean_deviations_w), '.b')
    axes.plot(data['time'],
              minmax_scaler_shift.transform(mean_deviations_w), 'b',
              label=label_w)

    axes2 = axes.twinx()
    minmax_scaler_flux = preprocessing.MinMaxScaler()
    axes2.plot(flux["time"], minmax_scaler_flux.fit_transform(flux["flux"]),
               'r', lw=2)
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
    axes.legend(loc='best')
    fig.tight_layout()
    canvas.print_figure(os.path.join(save_dir, '{}_average.png'.format(source)),
                        dpi=300)

