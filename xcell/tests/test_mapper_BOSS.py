import numpy as np
import xcell as xc
import healpy as hp


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog.fits'],
            'random_catalogs': ['xcell/tests/data/catalog.fits',
                                'xcell/tests/data/catalog.fits'],
            'z_edges': [0, 1.5], 'nside': 32, 'mask_name': 'mask',
            'coords': 'C', 'SDSS_name': 'dummy'}


def get_mapper(c=None):
    if c is None:
        c = get_config()
    return xc.mappers.MapperBOSS(c)


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    assert len(z) == 50
    assert len(nz) == 50
    LHS = (3/4)*16*hp.nside2npix(32)
    assert np.sum(nz) == LHS


def test_get_mask():
    m = get_mapper()
    d = m.get_mask()
    LHS = 12
    assert np.all(np.fabs(d-LHS) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    nl_pred = 2*hp.nside2npix(32)*(8**2+1*8**2)
    nl_pred *= pix_area**2/(4*np.pi)*(3/4)**2
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)
