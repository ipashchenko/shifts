import os
import sys
path = os.path.normpath('/home/ilya/github/vlbi_errors')
sys.path.insert(0, path)
from vlbi_errors.image_ops import rms_image_shifted
import urllib
import BeautifulSoup
import urllib2
import fnmatch
import pandas as pd
from utils import OrderedDefaultDict


mojave_bands = ['x', 'y', 'j', 'u', 'l18', 'l20', 'l21', 'l22']
l_bands = ['l18', 'l20', 'l21', 'l22']
# TODO: check connection to MOJAVE servers
mojave_multifreq_url = "http://www.cv.nrao.edu/2cmVLBA/data/multifreq/"
# Path to u-frequency file: dir/source/epoch/fname
mojave_u_url = "http://www.cv.nrao.edu/2cmVLBA/data/"
mojave_l_url = "http://www.cv.nrao.edu/MOJAVELBAND"
path_to_script = "/home/ilya/github/vlbi_errors/difmap/final_clean_nw"


def mojave_uv_fits_fname(source, band, epoch, ext='uvf'):
    return source + '.' + band + '.' + epoch + '.' + ext


def download_mojave_uv_fits(source, epochs=None, bands=None, download_dir=None):
    """
    Download FITS-files with self-calibrated uv-data from MOJAVE server.

    :param source:
        Source name [B1950].
    :param epochs: (optional)
        Iterable of epochs to download [YYYY-MM-DD]. If ``None`` then download
        all. (default: ``None``)
    :param bands: (optional)
        Iterable bands to download ('x', 'y', 'j' or 'u'). If ``None`` then
        download all available bands for given epochs. (default: ``None``)
    :param download_dir: (optional)
        Local directory to save files. If ``None`` then use CWD. (default:
        ``None``)
    """
    if bands is None:
        bands = ['u']
    else:
        assert set(bands).issubset(mojave_bands)

    if 'u' in bands:
        # Finding epochs in u-band data
        request = urllib2.Request(os.path.join(mojave_u_url, source))
        response = urllib2.urlopen(request)
        soup = BeautifulSoup.BeautifulSoup(response)

        available_epochs = list()
        for a in soup.findAll('a'):
            if fnmatch.fnmatch(a['href'], "*_*_*"):
                epoch = str(a['href'].strip('/'))
                available_epochs.append(epoch)

        if epochs is not None:
            if not set(epochs).issubset(available_epochs):
                raise Exception(" No epochs {} in MOJAVE data."
                                " Available are {}".format(epochs,
                                                           available_epochs))
        else:
            epochs = available_epochs

        # Downloading u-band data
        u_url = os.path.join(mojave_u_url, source)
        for epoch in epochs:
            fname = mojave_uv_fits_fname(source, 'u', epoch)
            url = os.path.join(u_url, epoch, fname)
            print("Downloading file {}".format(fname))
            path = os.path.join(download_dir, fname)
            if os.path.isfile(path):
                print("File {} does exist in {}."
                      " Skipping...".format(fname, download_dir))
                continue
            urllib.urlretrieve(url, path)

    # Downloading (optionally) x, y & j-band data
    request = urllib2.Request(mojave_multifreq_url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup.BeautifulSoup(response)

    download_list = list()
    for a in soup.findAll('a'):
        if source in a['href'] and '.uvf' in a['href']:
            fname = a['href']
            epoch = fname.split('.')[2]
            band = fname.split('.')[1]
            if band in bands:
                if epochs is None:
                    download_list.append(os.path.join(mojave_multifreq_url,
                                                      fname))
                else:
                    if epoch in epochs:
                        download_list.append(os.path.join(mojave_multifreq_url,
                                                          fname))
    for url in download_list:
        fname = os.path.split(url)[-1]
        print("Downloading file {}".format(fname))
        path = os.path.join(download_dir, fname)
        if os.path.isfile(path):
            print("File {} does exist in {}."
                  " Skipping...".format(fname, download_dir))
            continue
        urllib.urlretrieve(url, os.path.join(download_dir, fname))

    # Downloading (optionally) l-band data
    if 'l18' in bands or 'l20' in bands or 'l21' in bands or 'l22' in bands:
        request = urllib2.Request(os.path.join(mojave_l_url, source))
        try:
            response = urllib2.urlopen(request)
        except urllib2.HTTPError:
            print("No L-bands data available")
            return
        soup = BeautifulSoup.BeautifulSoup(response)

        available_epochs = list()
        for a in soup.findAll('a'):
            if fnmatch.fnmatch(a['href'], "*_*_*"):
                epoch = str(a['href'].strip('/'))
                available_epochs.append(epoch)

        if epochs is not None:
            if not set(epochs).issubset(available_epochs):
                raise Exception(" No epochs {} in MOJAVE data")
        else:
            epochs = available_epochs

        # Downloading l-band data
        l_url = os.path.join(mojave_l_url, source)
        for epoch in epochs:
            for band in bands:
                if band in l_bands:
                    fname = mojave_uv_fits_fname(source, band, epoch)
                    url = os.path.join(l_url, epoch, fname)
                    print("Downloading file {}".format(fname))
                    path = os.path.join(download_dir, fname)
                    if os.path.isfile(path):
                        print("File {} does exist in {}."
                              " Skipping...".format(fname, download_dir))
                        continue
                    urllib.urlretrieve(url, os.path.join(download_dir, fname))


def convert_fracyear_to_datetime(frac_year):
    """
    Adapted from here: https://stackoverflow.com/a/20911144

    :param frac_year:
        String of time in format ``YYYY.YYY...``
    :return:
        String of time in format ``YYYY_MM_DD``.
    """
    from datetime import datetime, timedelta
    year = int(frac_year)
    rem = frac_year - year

    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) -
                                       base).total_seconds() * rem)
    return "{}_{}_{}".format(result.year, str(result.month).zfill(2),
                             str(result.day).zfill(2))


if __name__ == '__main__':
    from pprint import pprint
    data_dir = "/home/ilya/github/shifts/data/mojave"
    source_epoch_file = "/home/ilya/github/shifts/data/mojave/mojave_source_epoch.txt"
    # Get pandas.DF from this file
    df = pd.read_table(source_epoch_file, delim_whitespace=True,
                       names=['source', 'epoch'], engine='python',
                       usecols=[0, 1])
    from collections import OrderedDict
    rms_values = OrderedDefaultDict(OrderedDict)
    for source, epoch in zip(df.source.values, df.epoch.values):
        epoch = convert_fracyear_to_datetime(epoch)
        try:
            print("Trying epoch {}".format(epoch))
            download_mojave_uv_fits(source, epochs=[epoch], bands=['u'],
                                    download_dir=data_dir)
        except:
            print("Failed")
            year = epoch.split('_')[0]
            month = epoch.split('_')[1]
            day = epoch.split('_')[2]
            epoch = '{}_{}_{}'.format(year, month, str(int(day)+1).zfill(2))
            print("Trying epoch {}".format(epoch))
            try:
                download_mojave_uv_fits(source, epochs=[epoch], bands=['u'],
                                        download_dir=data_dir)
            except:
                print("Failed")
                epoch = '{}_{}_{}'.format(year, month, str(int(day)-1).zfill(2))
                print("Trying epoch {}".format(epoch))
                try:
                    download_mojave_uv_fits(source, epochs=[epoch], bands=['u'],
                                            download_dir=data_dir)
                except:
                    print("Complete fail")
                    rms_values[source][epoch] = 0
                    continue

        uv_fits_fname = mojave_uv_fits_fname(source, band='u', epoch=epoch)
        uv_fits_path = os.path.join(data_dir, uv_fits_fname)
        rms = rms_image_shifted(uv_fits_path, hovatta_factor=False,
                                shift=(1000, 1000),
                                tmp_name='shifted_clean_map.fits',
                                tmp_dir=data_dir, stokes='I', image=None,
                                image_fits=None, mapsize_clean=(1024, 0.1),
                                path_to_script=path_to_script, niter=None)
        rms_values[source][epoch] = rms
        pprint(rms_values)