import xcell as xc
import numpy as np


def get_config():
    return {'file_klm': 'xcell/tests/data/alm.fits',
            'file_mask': 'xcell/tests/data/map.fits',
            'file_noise': 'xcell/tests/data/nl.txt',
            'mask_name': 'mask_CMBK',
            'mask_aposize': 3,  # Must be large than pixel size
            'mask_apotype': 'C1',
            'nside': 32}


def get_mapper():
    config = get_config()
    return xc.mappers.MapperP18CMBK(config)


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


def test_get_nl_coupled():
    m = get_mapper()
    nl = m.get_nl_coupled()
    cl = m.get_cl_fiducial()
    ll_tb = m.get_ells_in_table()
    ell = m.get_ell()

    assert nl.shape == (1, 3*32)
    assert np.all(np.fabs(nl) < 1E-15)
    assert cl.shape == (1, 3*32)
    assert np.allclose(ll_tb, np.arange(3*32))
    assert np.all(ell == np.arange(3 * 32))


def test_get_nmt_field():
    import pymaster as nmt
    m = get_mapper()
    f = m.get_nmt_field()
    cl = nmt.compute_coupled_cell(f, f)[0]
    assert np.fabs(cl[0]-4*np.pi) < 1E-3
    assert np.all(np.fabs(cl[1:]) < 1E-5)
