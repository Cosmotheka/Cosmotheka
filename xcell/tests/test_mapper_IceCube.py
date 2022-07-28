import numpy as np
import healpy as hp
from astropy.table import Table
import xcell as xc


def get_config():
    return {'EventDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/ICEvents',
            'FakeEventDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/FakeICEvents',
            'AeffDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/ICAeff',
            'FakeAeffDir': '/Users/pattersonb/Documents/Internship_Project' +
            '/xCell/xcell/tests/data/FakeICAeff',
            'nside': 128,
            'coords': 'G'}


def make_fake_data():
    config = get_config()
    FakeEventDir = config['FakeEventDir']
    FakeAeffDir = config['FakeAeffDir']
    seasons = ['40', '59', '79', '86_I', '86_II',
               '86_III', '86_IV', '86_V', '86_VI', '86_VII']
    nside = config['nside']
    npix = hp.nside2npix(nside)
    # one event per pixel (per season per energy bin)
    r_g2c = hp.Rotator(coord=['G', 'C'])
    ra, dec = r_g2c(hp.pix2ang(nside, np.arange(npix), lonlat=True), lonlat = True)
    for i, season in enumerate(seasons):
        # generates fake event data
        logEcol = [2.01]*len(ra) + [3.01]*len(ra) + [4.01]*len(ra) + [5.01]*len(ra)
        racol = list(ra)*4
        deccol = list(dec)*4
        e = Table({'log10(E/GeV)': logEcol,
                   'RA[deg]': racol,
                   'Dec[deg]': deccol})
        e.write(f'{FakeEventDir}/IC{season}_exp.csv', overwrite=True, 
                delimiter='\t')

        # generates fake aeff data with aeff = 1 everywhere
        if i <= 4:
            Emin = np.array(list(np.arange(40)/5 + 2) * 50).round(1)
            Emax = (Emin + 0.2).round(1)
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
            a.write(f'{FakeAeffDir}/IC{season}_effectiveArea.csv', overwrite=True,
                    delimiter='\t')


def test_get_events():
    make_fake_data()
    config = get_config()
    config['EventDir'] = config['FakeEventDir']
    mapper = xc.mappers.MapperIceCube(config)
    for i in range(10):
        cats = mapper._get_events(i)
        for j in cats:
            assert len(j) == hp.nside2npix(config['nside'])


def test_get_aeff_mask():
    config = get_config()
    config['AeffDir'] = config['FakeAeffDir']
    mapper = xc.mappers.MapperIceCube(config)
    for i in range(10):
        Aeff = mapper._get_aeff(i)
        for j in range(4):
            AeffMask, AeffMap = mapper._get_aeff_mask(Aeff[j])
            assert (len(np.where(AeffMask == 0)[0]) +
                    len(np.where(AeffMask == 1)[0]) == len(AeffMask))
            assert (len(np.where(np.fabs(AeffMap) < 1E-15)[0]) +
                    len(np.where(np.fabs(AeffMap - 1) < 1E-15)[0]) == len(AeffMap))
            assert (len(np.where(AeffMask == 0)[0]) ==
                    len(np.where(np.fabs(AeffMap) < 1E-15)[0]))
            assert len(AeffMask) == len(AeffMap)


def test_get_mask():
    config = get_config()
    mapper = xc.mappers.MapperIceCube(config)
    mask = mapper.get_mask()
    assert len(mask) == hp.nside2npix(config['nside'])
    assert (len(np.where(mask == 0)[0]) +
           len(np.where(mask == 1)[0]) == len(mask))


def test_get_signal_map():
    config = get_config()
    npix = hp.nside2npix(config['nside'])
    mapper = xc.mappers.MapperIceCube(config)
    maps = []
    for i in range(4):
        maps.append(mapper.get_signal_map(i))
    assert np.shape(maps) == (4, npix)
    assert np.max(maps[0]) == 5.19153
    assert np.max(maps[1]) == 5.53781
    assert np.max(maps[2]) == 59.3064
    assert np.max(maps[3]) == 1112.74


def test_get_signal_map_fakeAeff():
    config = get_config()
    config['EventDir'] = config['FakeEventDir']
    config['AeffDir'] = config['FakeAeffDir']
    mapper = xc.mappers.MapperIceCube(config)
    for i in range(4):
        map = mapper.get_signal_map(i)
        assert np.all(np.fabs(map) < 1E-15)
