import xcell as xc
import numpy as np
import pytest


def get_config():
    c = {'file_map': 'xcell/tests/data/map.fits',
         'file_hm1': 'xcell/tests/data/hm1_map.fits',
         'file_hm2': 'xcell/tests/data/hm2_map.fits',
         'file_mask': 'xcell/tests/data/map.fits',
         'file_gp_mask': 'xcell/tests/data/mask1.fits',
         'file_sp_mask': 'xcell/tests/data/mask2.fits',
         'gal_mask_mode': '0.2',
         'nside': 32}
    return c


@pytest.mark.parametrize('m', [xc.mappers.MapperP15tSZ(get_config()),
                               xc.mappers.MapperP18SMICA(get_config()),
                               xc.mappers.MapperP15CIB(get_config())])
def test_spin(m):
    assert m.get_spin() == 0


@pytest.mark.parametrize('m', [xc.mappers.MapperP15tSZ(get_config()),
                               xc.mappers.MapperP18SMICA(get_config()),
                               xc.mappers.MapperP15CIB(get_config())])
def test_get_signal_map(m):
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d-1) < 0.02)


@pytest.mark.parametrize('m', [xc.mappers.MapperP15tSZ(get_config()),
                               xc.mappers.MapperP18SMICA(get_config()),
                               xc.mappers.MapperP15CIB(get_config())])
def test_get_nl_coupled(m):
    nl = m.get_nl_coupled()
    assert np.mean(nl) < 0.001


@pytest.mark.parametrize(['m', 'n'],
                         [[xc.mappers.MapperP15tSZ(get_config()),
                           xc.mappers.MapperP15tSZ(get_config())],
                         [xc.mappers.MapperP18SMICA(get_config()),
                          xc.mappers.MapperP18SMICA(get_config())],
                         [xc.mappers.MapperP15CIB(get_config()),
                          xc.mappers.MapperP15CIB(get_config())]])
def test_get_cl_coupled(m, n):
    m.file_map = 'xcell/tests/data/map_auto_test.fits'
    m.cl_mode = 'Auto'
    n.file_map = 'xcell/tests/data/map_auto_test.fits'
    n.cl_mode = 'Cross'
    nl = m.get_nl_coupled()
    cl = m.get_cl_coupled()
    cl_signal = n.get_cl_coupled()
    assert nl.shape == (1, 3*32)
    assert cl.shape == (1, 3*32)
    assert cl_signal.shape == (1, 3*32)
    assert abs(np.mean((cl[0]-nl[0]-cl_signal[0])[20:])) < 0.001
