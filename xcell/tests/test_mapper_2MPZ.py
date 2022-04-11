import numpy as np
import xcell as xc
import healpy as hp
import os
import pytest


def get_config():
    return {'data_catalog': 'xcell/tests/data/catalog_2mpz.fits',
            'mask': 'xcell/tests/data/map.fits',
            'z_edges': [-1E-10, 0.5],
            'path_rerun': '.',
            'coords': 'C',
            'nside': 32, 'mask_name': 'mask'}


def cleanup_rerun():
    for fname in ['nz_2MPZ.npz',
                  'mask_mask_coordC_ns32.fits.gz',
                  'mask_mask_coordG_ns32.fits.gz']:
        if os.path.isfile(fname):
            os.remove(fname)


def get_mapper():
    return xc.mappers.Mapper2MPZ(get_config())


def test_smoke():
    cleanup_rerun()
    m = get_mapper()
    m.get_catalog()
    assert len(m.cat_data) == hp.nside2npix(32)
    cleanup_rerun()


def test_get_nz():
    cleanup_rerun()
    m = get_mapper()
    m.get_catalog()
    z, nz = m.get_nz()
    h, b = np.histogram(m.cat_data['ZSPEC'],
                        range=[0.0, 0.4], bins=100,
                        density=True)
    z_arr = 0.5 * (b[:-1] + b[1:])
    assert np.all(np.fabs(z-z_arr) < 1E-5)
    assert np.all(np.fabs((nz-h)/np.amax(nz)) < 1E-3)

    # Read from file and compare again
    assert os.path.isfile('nz_2MPZ.npz')
    m = get_mapper()
    m.get_catalog()
    z2, nz2 = m.get_nz()
    assert np.all(np.fabs(nz2-nz) < 1E-5)
    cleanup_rerun()


@pytest.mark.parametrize('coord', ['G', 'C', 'E'])
def test_get_signal_map(coord):
    cleanup_rerun()
    c = get_config()
    c['coords'] = coord
    try:
        m = xc.mappers.Mapper2MPZ(c)
    except NotImplementedError:
        assert coord == 'E'
        return
    d = m.get_signal_map()
    d = np.array(d)
    assert d.shape == (1, hp.nside2npix(m.nside))
    assert np.all(np.fabs(d) < 1E-15)
    cleanup_rerun()


def test_get_mask():
    cleanup_rerun()
    m = get_mapper()
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)
    cleanup_rerun()


def test_get_nl_coupled():
    cleanup_rerun()
    m = get_mapper()
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    nl_pred = hp.nside2npix(32)
    nl_pred *= pix_area**2/(4*np.pi)
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)
    cleanup_rerun()


def test_get_dtype():
    m = get_mapper()
    assert m.get_dtype() == 'galaxy_density'


def test_get_spin():
    m = get_mapper()
    assert m.get_spin() == 0
