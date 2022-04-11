import numpy as np
import xcell as xc
import healpy as hp
import os
from astropy.io import fits
from astropy.table import Table
import pytest


def get_config():
    return {'data_catalog': 'xcell/tests/data/catalog_2mpz.fits',
            'mask': 'xcell/tests/data/map.fits',
            'star_map': 'xcell/tests/data/map.fits',
            'spec_sample': 'xcell/tests/data/catalog_spec_2mpz.csv',
            'z_edges': [-1E-10, 0.5],
            'bin_name': '0',
            'path_rerun': '.',
            'coords': 'C',
            'apply_galactic_correction': False,
            'nside': 32, 'mask_name': 'mask'}


def cleanup_rerun():
    for fname in ['nz_WIxSC_bin0.npz', 'WIxSC_rerun_bin0.fits',
                  'mask_mask_coordC_ns32.fits.gz',
                  'mask_mask_coordG_ns32.fits.gz']:
        if os.path.isfile(fname):
            os.remove(fname)


def get_mapper():
    return xc.mappers.MapperWIxSC(get_config())


def test_smoke():
    cleanup_rerun()
    m = get_mapper()
    cat = m.get_catalog()
    assert len(m.cat_data) == hp.nside2npix(32)
    # Check that the rerun catalog has been created
    assert os.path.isfile('./WIxSC_rerun_bin0.fits')
    t = Table.read('./WIxSC_rerun_bin0.fits')
    assert (cat['RA'] == t['RA']).all()


def test_get_nz():
    cleanup_rerun()
    m = get_mapper()
    m.get_catalog()
    z, nz = m.get_nz()
    with fits.open('xcell/tests/data/catalog_2mpz.fits') as f:
        cat = Table.read(f, format='fits', memmap=True)
    h, b = np.histogram(cat['ZPHOTO_CORR'],
                        range=[0.0, 0.6], bins=150,
                        density=True)
    z_arr = 0.5 * (b[:-1] + b[1:])
    assert np.all(np.fabs(z-z_arr) < 1E-5)
    assert np.all(np.fabs((nz-h)/np.amax(nz)) < 1E-3)

    # Read from file and compare again
    assert os.path.isfile('nz_WIxSC_bin0.npz')
    m = get_mapper()
    m.get_catalog()
    z2, nz2 = m.get_nz()
    assert np.all(np.fabs(nz2-nz) < 1E-5)


@pytest.mark.parametrize('coord', ['G', 'C', 'E'])
def test_get_signal_map(coord):
    cleanup_rerun()
    c = get_config()
    c['coords'] = coord
    try:
        m = xc.mappers.MapperWIxSC(c)
    except NotImplementedError:
        assert coord == 'E'
        return
    d = m.get_signal_map()
    d = np.array(d)
    assert d.shape == (1, hp.nside2npix(m.nside))
    assert np.all(np.fabs(d) < 1E-15)


def test_get_mask():
    cleanup_rerun()
    m = get_mapper()
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)


def test_galactic_correction():
    # Test that the galactic corrector returns something close
    # to zero when using a map of stars that is the same as
    # the signal.
    np.random.seed(1234)
    cleanup_rerun()
    m = get_mapper()
    nside = 32
    npix = hp.nside2npix(nside)
    delta = np.random.randn(npix)
    stars = 10.**delta
    mask = np.ones(npix)
    d = m._get_galactic_correction(delta, stars, mask)
    assert np.std(delta-d['delta_map']) < 0.1


def test_get_nl_coupled():
    cleanup_rerun()
    m = get_mapper()
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    nl_pred = hp.nside2npix(32)
    nl_pred *= pix_area**2/(4*np.pi)
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)


def test_get_dtype():
    cleanup_rerun()
    m = get_mapper()
    assert m.get_dtype() == 'galaxy_density'


def test_get_spin():
    m = get_mapper()
    assert m.get_spin() == 0


def test_get_nl_coupled_data():
    c = get_config()
    c['nside_nl_threshold'] = 1
    c['lmin_nl_from_data'] = 10
    c['nl_analytic'] = False
    m = xc.mappers.MapperWIxSC(c)
    nl = m.get_nl_coupled()
    assert np.all(nl == 0)
    cleanup_rerun()
