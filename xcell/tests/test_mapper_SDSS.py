import numpy as np
import xcell as xc
import healpy as hp
import pytest
import os


def get_config():
    c = {'data_catalogs': ['xcell/tests/data/catalog.fits',
                           'xcell/tests/data/catalog.fits'],
         'random_catalogs': ['xcell/tests/data/catalog.fits',
                             'xcell/tests/data/catalog.fits'],
         'z_edges': [0, 1.5], 'nside': 32, 'mask_name': 'mask'}
    c['nside_nl_threshold'] = 1
    c['lmin_nl_from_data'] = 10
    c['SDSS_name'] = 'dummy'
    c['coords'] = 'C'
    return c


@pytest.mark.parametrize('m', [xc.mappers.MappereBOSS(get_config()),
                               xc.mappers.MapperBOSS(get_config())])
def test_get_dtype(m):
    assert m.get_dtype() == 'galaxy_density'


@pytest.mark.parametrize('m', [xc.mappers.MappereBOSS(get_config()),
                               xc.mappers.MapperBOSS(get_config())])
def test_get_spin(m):
    assert m.get_spin() == 0


@pytest.mark.parametrize('m', [xc.mappers.MappereBOSS(get_config()),
                               xc.mappers.MapperBOSS(get_config())])
def test_smoke(m):
    m._get_w(mod='data')
    assert len(m.ws['data']) == 2*hp.nside2npix(32)


def test_error_base():
    with pytest.raises(NotImplementedError):
        xc.mappers.MapperSDSS({})


@pytest.mark.parametrize('m', [xc.mappers.MappereBOSS(get_config()),
                               xc.mappers.MapperBOSS(get_config())])
def test_get_signal_map(m):
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert len(d) == hp.nside2npix(m.nside)
    assert np.all(np.fabs(d) < 1E-15)


@pytest.mark.parametrize('m', [xc.mappers.MappereBOSS(get_config()),
                               xc.mappers.MapperBOSS(get_config())])
def test_get_nl_coupled_data(m):
    nl = m.get_nl_coupled()
    assert np.all(nl == 0)


def test_rerun():
    fname_msk = 'xcell/tests/data/mask_mask_coordC_ns32.fits.gz'
    fname_map = 'xcell/tests/data/SDSS_dummy_signal_coordC_ns32.fits.gz'
    fname_nls = 'xcell/tests/data/SDSS_dummy_Nell_coordC_ns32.npz'

    # Cleanup just in case
    if os.path.isfile(fname_msk):
        os.remove(fname_msk)
    if os.path.isfile(fname_map):
        os.remove(fname_map)

    c = get_config()
    c['path_rerun'] = 'xcell/tests/data/'
    m1 = xc.mappers.MapperBOSS(c)
    msk1 = m1.get_mask()
    map1 = m1.get_signal_map()[0]
    nl1 = m1.get_nl_coupled()

    # Check maps exist
    assert os.path.isfile(fname_msk)
    assert os.path.isfile(fname_map)
    assert os.path.isfile(fname_nls)

    # Now they will be read from file
    m2 = xc.mappers.MapperBOSS(c)
    msk2 = m2.get_mask()
    map2 = m2.get_signal_map()[0]
    nl2 = m2.get_nl_coupled()

    # Check the maps are the same
    assert np.all(map1-map2 == 0.0)
    assert np.all(msk1-msk2 == 0.0)
    assert np.all(nl1-nl2 == 0.0)

    # Final cleanup
    os.remove(fname_msk)
    os.remove(fname_map)
    os.remove(fname_nls)
