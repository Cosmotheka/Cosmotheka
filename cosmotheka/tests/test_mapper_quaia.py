import numpy as np
import cosmotheka as xc
import healpy as hp
import os


def get_config():
    return {'data_catalog': 'cosmotheka/tests/data/catalog.fits',
            'selection_G': 'cosmotheka/tests/data/map.fits',
            'selection_C': 'cosmotheka/tests/data/map.fits',
            'mask_extra_G': 'cosmotheka/tests/data/map.fits',
            'mask_extra_C': 'cosmotheka/tests/data/map.fits',
            'z_name': 'Z',
            'z_edges': [-1E-10, 1.0],
            'path_rerun': '.',
            'coords': 'C',
            'nside': 32, 'mask_name': 'mask'}


def cleanup_rerun():
    for fname in ['Quaia_z-0.000_1.000_cat.fits',
                  'Quaia_z-0.000_1.000_dndz.npz',
                  'mask_mask_coordC_ns32.fits.gz',
                  'Quaia_signal_map_coordC_ns32.fits.gz']:
        if os.path.isfile(fname):
            os.remove(fname)


def get_mapper():
    return xc.mappers.MapperQuaia(get_config())


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
    nz /= np.amax(nz)
    nz_test = np.exp(-0.5*((z-0.59)/0.1)**2)
    assert np.all(np.fabs(nz-nz_test) < 1E-3)

    # Read from file and compare again
    assert os.path.isfile('Quaia_z-0.000_1.000_dndz.npz')
    m = get_mapper()
    m.get_catalog()
    z2, nz2 = m.get_nz()
    nz2 /= np.amax(nz2)
    assert np.all(np.fabs(nz2-nz) < 1E-5)
    cleanup_rerun()


def test_get_signal_map():
    cleanup_rerun()
    c = get_config()
    m = xc.mappers.MapperQuaia(c)
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
