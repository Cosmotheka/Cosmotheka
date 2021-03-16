import numpy as np
import xcell as xc
import healpy as hp


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog.fits'],
            'completeness_map': 'xcell/tests/data/map.fits',
            'binary_mask': 'xcell/tests/data/map.fits',
            'z_arr_dim': 500,
            'star_map': 'xcell/tests/data/map.fits',
            'zbin': 0, 'nside': 32, 'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperDECaLS(get_config())


def test_smoke():
    m = get_mapper()
    m.get_catalogs()
    assert len(m.cat_data) == 2*hp.nside2npix(32)


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    h, b = np.histogram(m.cat_data[m.pz][m.mskflag],
                        range=[-0.3, 1], bins=m.z_arr_dim)
    z_arr = 0.5 * (b[:-1] + b[1:])
    sel = z_arr > 0
    assert len(z) == len(z_arr[sel])
    assert len(nz) == len(z_arr[sel])
    z_hpeak = z_arr[np.where(h == max(h))[0][0]]
    z_nzpeak = z[np.where(nz == max(nz))[0][0]]
    assert z_hpeak == z_nzpeak


def test_lorentzian():
    m = get_mapper()
    assert m._get_lorentzian(np.array([0])) == 0.99665079001703


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map(apply_galactic_correction=False)
    assert len(d) == 1
    d = d[0]
    assert len(d) == hp.nside2npix(m.nside)
    assert np.all(np.fabs(d) < 1E-15)


def test_get_mask():
    m = get_mapper()
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    nl_pred = hp.nside2npix(32)/2
    nl_pred *= pix_area**2/(4*np.pi)
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)
