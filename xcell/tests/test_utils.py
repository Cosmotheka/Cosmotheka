import xcell as xc
import healpy as hp
import numpy as np


def test_map_from_points():
    nside = 32
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside,
                         np.arange(npix),
                         lonlat=True)
    m = xc.mappers.get_map_from_points({'RA': ra,
                                        'DEC': dec},
                                       nside)
    assert np.all(m == 1)


def test_get_DIR_Nz():
    # If cat_spec and cat_photo are the same,
    # DIR should return the N(z) of the spec catalog.
    cat = {'z': np.random.randn(1000),
           'rmag': 2+np.random.rand(1000),
           'imag': 2+np.random.rand(1000)}
    z, nz, nz_jk = xc.mappers.get_DIR_Nz(cat, cat,
                                         ['rmag', 'imag'],
                                         'z', [-3, 3], 10,
                                         bands_photo=['rmag', 'imag'])

    nzz, ze = np.histogram(cat['z'], range=[-3, 3], bins=10, density=True)
    assert np.all((nzz-nz)/np.amax(nzz) < 1E-10)
