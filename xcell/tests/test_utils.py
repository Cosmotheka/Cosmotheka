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
