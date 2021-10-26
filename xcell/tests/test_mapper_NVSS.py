import numpy as np
import xcell as xc
import healpy as hp
import os
from astropy.table import Table


def get_config():
    return {'data_catalog': 'xcell/tests/data/catalog_nvss.fits',
            'mask_sources': 'xcell/tests/data/source_mask_nvss.txt',
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
    
    
    mask = m.get_mask()
    signal = m.get_signal_map()
    
    
    assert len(c) < hp.nside2npix(32)
    assert len(mask) < hp.nside2npix(32)
    assert len(signal) < hp.nside2npix(32)
    
    clean_fake_data()