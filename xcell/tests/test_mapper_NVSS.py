import numpy as np
import xcell as xc
import healpy as hp
import os
from astropy.table import Table


def get_config():
    return {'data_catalog': 'xcell/tests/data/catalog_nvss.fits',
            'mask_sources': 'xcell/tests/data/source_masks_nvss.txt',
            'nside': 32, 'mask_name': 'mask'}


def make_fake_data():
    nside = 32
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    flux = 10+(1000-10)*np.random.rand(npix)
    c = Table({'RAJ2000': ra,
               'DEJ2000': dec,
               'S1_4': flux})
    c.write("xcell/tests/data/catalog_nvss.fits", overwrite=True)


def clean_fake_data():
    os.remove("xcell/tests/data/catalog_nvss.fits")


def test_basic():
    make_fake_data()
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    c = m.get_catalog()
    assert len(c) < hp.nside2npix(32)
    clean_fake_data()


def test_get_mask():
    config = get_config()
    config['DEC_min_deg'] = -90.
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperNVSS(config)
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)


def test_get_signal_map():
    make_fake_data()
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    d = m.get_signal_map()
    d = np.array(d)
    assert d.shape == (1, hp.nside2npix(m.nside))
    assert np.all(np.fabs(d) < 1E-15)
    clean_fake_data()


def test_get_dtype():
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    assert m.get_dtype() == 'galaxy_density'


def test_get_spin():
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    assert m.get_spin() == 0
