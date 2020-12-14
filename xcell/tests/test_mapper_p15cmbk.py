import xcell as xc
import numpy as np


def get_config():
    return {'file_klm': 'xcell/tests/data/alm.fits',
            'file_mask': 'xcell/tests/data/map.fits',
            'file_noise': 'xcell/tests/data/nl.txt',
            'mask_name': 'mask_CMBK',
            'nside': 32}


def get_mapper():
    config = get_config()
    return xc.mappers.MapperP15CMBK(config)


def test_smoke():
    get_mapper()


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d-1) < 0.02)


def test_get_mask():
    m = get_mapper()
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)


def test_get_nl():
    m = get_mapper()
    nl = m.get_nl()
    cl = m.get_cl_fiducial()
    ll = m.get_ells()
    assert nl.shape == (1, 3*32)
    assert np.all(np.fabs(nl) < 1E-15)
    assert cl.shape == (1, 3*32)
    assert np.allclose(ll, np.arange(3*32))
