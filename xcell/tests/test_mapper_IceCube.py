import numpy as np
import healpy as hp
from astropy.table import Table
import xcell as xc


def get_config():
    return {'EventDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/ICEvents',
            'AeffDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/ICAeff',
            'FakeAeffDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/FakeICAeff',
            'nside': 128,
            'coords': 'G'}


def make_fake_data():
    config = get_config()
    EventDir = config['EventDir']
    FakeAeffDir = config['FakeAeffDir']
    seasons = ['40', '59', '79', '86_I', '86_II',
               '86_III', '86_IV', '86_V', '86_VI', '86_VII']
    nside = config['nside']
    npix = hp.nside2npix(nside)
    # one event per pixel
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    for i in seasons:
        # generates fake event data
        logE = 2.01 + (3.98*np.random.rand(npix))
        e = Table({'log10(E/GeV)': logE,
                   'RA[deg]': ra,
                   'Dec[deg]': dec})
        e.write(f'{EventDir}/IC{i}_exp.csv', overwrite=True)

        # generates fake aeff data with aeff = 1 everywhere
        Emin = list(np.arange(40)/5 + 2) * 50
        Emax = Emin + 0.2
        DecColBase = [-90.0, -73.74, -66.93, -61.64, -57.14,
                      -53.13, -49.46, -46.05, -42.84, -39.79,
                      -36.87, -34.06, -31.33, -28.69, -26.1,
                      -23.58, -21.1, -18.66, -16.26, -13.89,
                      -11.54, -9.21, -6.89, -4.59, -2.29, 0.0,
                      2.29, 4.59, 6.89, 9.21, 11.54, 13.89,
                      16.26, 18.66, 21.1, 23.58, 26.1, 28.69,
                      31.33, 34.06, 36.87, 39.79, 42.84, 46.05,
                      49.46, 53.13, 57.14, 61.64, 66.93, 73.74,
                      90.0]
        DecMin = np.repeat(DecColBase[:-1], 40)
        DecMax = np.repeat(DecColBase[1:], 40)
        AeffCol = list([10**4])*(40*50)
        a = Table({'log10(E_nu/GeV)_min': Emin,
                   'log10(E_nu/GeV)_max': Emax,
                   'Dec_nu_min[deg]': DecMin,
                   'Dec_nu_max[deg]': DecMax,
                   'A_Eff[cm^2]': AeffCol})
        a.write(f'{FakeAeffDir}/IC{i}_effectiveArea.csv', overwrite=True)


def test_get_events():
    make_fake_data()
    config = get_config()
    mapper = xc.mappers.MapperIceCube(config)
    for i in range(10):
        cats = mapper._get_events(i)
        totalevents = 0
        for j in cats:
            totalevents += len(j)
        assert totalevents == hp.nside2npix(config['nside'])


def test_get_mask():
    config = get_config()
    mapper = xc.mappers.MapperIceCube(config)
    mask = mapper.get_mask()
    assert len(mask) == hp.nside2npix(config['nside'])
    assert (len(np.where(mask == 0)[0]) +
           len(np.where(mask == 1)[0]) == len(mask))


def test_get_signal_map():
    make_fake_data()
    config = get_config()
    npix = hp.nside2npix(config['nside'])
    mapper = xc.mappers.MapperIceCube(config)
    maps = mapper.get_signal_map()
    assert np.shape(maps) == (4, npix)
    assert np.all(np.fabs(maps[0]) < 1E-15)
