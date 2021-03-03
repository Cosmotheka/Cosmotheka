import numpy as np
import xcell as xc
import healpy as hp


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog.fits'],
            'comp_map': 'xcell/tests/data/map.fits',
            'mask': 'xcell/tests/data/map.fits',
            'stars': 'xcell/tests/data/map.fits',
            'z_bin': 0, 'nside': 32, 'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperDECaLS(get_config())


def test_smoke():
    m = get_mapper()
    assert len(m.w_data) == 2*hp.nside2npix(32)


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    assert len(z) == 50
    assert len(nz) == 50
    assert np.sum(nz) == 16*hp.nside2npix(32)


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert len(d) == hp.nside2npix(m.nside)
    assert np.all(np.fabs(d) < 1E-15)


def test_get_mask():
    m = get_mapper()
    d = m.get_mask()
    assert np.all(np.fabs(d-16) < 1E-5)

def test_lorentzian():
    # TO DO 
    assert 0 == 0 

def test_get_nl_coupled():
    m = get_mapper()
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    # sum(w_d^2 + alpha^2 * w_r^2)
    nl_pred = 2*hp.nside2npix(32)*(8**2+1*8**2)
    nl_pred *= pix_area**2/(4*np.pi)
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)
