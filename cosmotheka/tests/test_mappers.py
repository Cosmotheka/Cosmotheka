import cosmotheka as xc
import pytest


class MapperBaseFunctional(xc.mappers.MapperBase):
    """ Just a class we create here with valid
    dtype and spin so we can test the base class'
    init functions.
    """
    dtype = 'generic'
    spin = 0


def test_mapper_from_name():
    for nm in ['MapperP18CMBK', 'MappereBOSS']:
        xc.mappers.mapper_from_name(nm)

    with pytest.raises(ValueError):
        xc.mappers.mapper_from_name('MapperPL15CMBK')


def test_maper_base_defaults():
    with pytest.raises(ValueError):
        # This should fail as base class should never be
        # initialised.
        xc.mappers.MapperBase(
            {'mask_name': 'mask', 'nside': 32, 'coords': 'C'})

    m = MapperBaseFunctional(
        {'mask_name': 'mask', 'nside': 32, 'coords': 'C'})

    assert m.nside == 32
    assert m.coords == 'C'
    assert m.mask_name == 'mask'
    assert m.get_spin() == 0
    assert m.get_dtype() == 'generic'

    with pytest.raises(NotImplementedError):
        m.get_signal_map()

    with pytest.raises(NotImplementedError):
        m.get_mask()

    with pytest.raises(NotImplementedError):
        m.get_nl_coupled()

    with pytest.raises(NotImplementedError):
        m.get_nl_covariance()

    with pytest.raises(NotImplementedError):
        m._get_nmt_catalog_field()
