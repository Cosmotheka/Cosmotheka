import numpy as np
import xcell as xc
import healpy as hp
from astropy.table import Table
import os


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog.fits'],
            'completeness_map': 'xcell/tests/data/map.fits',
            'binary_mask': 'xcell/tests/data/map.fits',
            'num_z_bins': 500, 'coords': 'C',
            'star_map': 'xcell/tests/data/map.fits',
            'zbin': 0, 'nside': 32, 'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperDELS(get_config())


def test_smoke():
    m = get_mapper()
    m.get_catalog()
    assert len(m.cat_data) == 2*hp.nside2npix(32)


def test_rerun():
    conf = get_config()
    conf['path_rerun'] = 'xcell/tests/data/'
    m = xc.mappers.MapperDELS(conf)
    cat = m.get_catalog()
    fn = 'xcell/tests/data/DELS_cat_bin0.fits'
    catb = Table.read(fn)
    assert len(catb) == len(cat)
    os.remove(fn)

    dndz = m.get_nz()
    fn = 'xcell/tests/data/DELS_dndz_bin0.npz'
    d = np.load(fn)
    assert (dndz[0] == d['z_mid'][d['z_mid'] >= 0]).all()
    assert (dndz[1] == d['nz'][d['z_mid'] >= 0]).all()
    os.remove(fn)


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    h, b = np.histogram(m.cat_data[m.pz][m.mskflag],
                        range=[-0.3, 1], bins=m.num_z_bins)
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
    d = np.array(d)
    assert d.shape == (1, hp.nside2npix(m.nside))
    assert np.all(np.fabs(d) < 1E-15)


def test_galactic_correction():
    # Test that the galactic corrector returns something close
    # to zero when using a map of stars that is the same as
    # the signal.
    np.random.seed(1234)
    m = get_mapper()
    nside = 32
    npix = hp.nside2npix(nside)
    delta = np.random.randn(npix)
    stars = 10.**delta
    mask = np.ones(npix)
    d = m._get_galactic_correction(delta, stars, mask)
    assert np.std(delta-d['delta_map']) < 0.1


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


def test_get_dtype():
    m = get_mapper()
    assert m.get_dtype() == 'galaxy_density'
