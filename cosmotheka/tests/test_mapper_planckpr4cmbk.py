import cosmotheka as xc
import numpy as np
import pytest


def get_config():
    return {'file_klm': 'cosmotheka/tests/data/alm.fits',
            'file_mask': 'cosmotheka/tests/data/map.fits',
            'file_noise': 'cosmotheka/tests/data/nl_pr4.txt',
            'mask_name': 'mask_CMBK',
            'mask_aposize': 3,  # Must be large than pixel size
            'mask_apotype': 'C1',
            'nside': 32, 'coords': 'C'}


def get_mapper():
    config = get_config()
    return xc.mappers.MapperPlanckPR4CMBK(config)


def test_get_nl_coupled():
    m = get_mapper()
    nl = m.get_nl_coupled()
    ell = m.get_ell()

    assert nl.shape == (1, 3*32)
    nl = nl.squeeze()
    assert np.all(np.fabs(nl[0]) < 1E-15)
    assert np.all(np.fabs(nl-1)[1:] < 1E-15)
    assert np.all(ell == np.arange(3 * 32))

    with pytest.raises(NotImplementedError):
        m.get_cl_fiducial()
