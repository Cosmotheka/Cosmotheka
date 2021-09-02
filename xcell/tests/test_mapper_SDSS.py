import numpy as np
import xcell as xc
import healpy as hp
import pytest


def get_config():
    c = {'data_catalogs': ['xcell/tests/data/catalog.fits',
                           'xcell/tests/data/catalog.fits'],
         'random_catalogs': ['xcell/tests/data/catalog.fits',
                             'xcell/tests/data/catalog.fits'],
         'z_edges': [0, 1.5], 'nside': 32, 'mask_name': 'mask'}
    c['nside_nl_threshold'] = 1
    c['lmin_nl_from_data'] = 10
    c['SDSS_name'] = 'dummy'
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
