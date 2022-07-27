import numpy as np
import healpy as hp
import os
from astropy.table import Table
import xcell as xc
import shutil


def get_config():
    return {'EventDir': '/Users/pattersonb/Documents/Internship_Project/xCell/xcell/tests/data/ICEvents',
            'AeffDir': '/Users/pattersonb/Documents/Internship_Project/xCell/xcell/tests/data/ICAeff',
            'nside': 128,
            'coords': 'G'}


def make_fake_data():
    config = get_config()
    EventDir = config['EventDir']
    seasons = ['40', '59', '79', '86_I', '86_II',
               '86_III', '86_IV', '86_V', '86_VI', '86_VII']
    nside = config['nside']
    npix = hp.nside2npix(nside)
    # one event per pixel
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    for i in seasons:
        # generates fake event data
        logE = 2.01 + (3.98*np.random.rand(npix))
        c = Table({'log10(E/GeV)': logE,
                   'RA[deg]': ra,
                   'Dec[deg]': dec})
        c.write(f'{EventDir}/IC{i}_exp.csv', overwrite=True)


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
    assert len(np.where(mask == 0)[0]) + len(np.where(mask == 1)[0]) == len(mask)


def test_get_signal_map():
    make_fake_data()
    config = get_config()
    npix = hp.nside2npix(config['nside'])
    mapper = xc.mappers.MapperIceCube(config)
    maps = mapper.get_signal_map()
    assert np.shape(maps) == (4, npix)
    assert np.all(np.fabs(maps[0]) < 1E-15)
