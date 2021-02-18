import xcell as xc
import pytest


def test_mapper_from_name():
    for nm in ['MapperP18CMBK', 'MappereBOSSQSO']:
        xc.mappers.mapper_from_name(nm)

    with pytest.raises(ValueError):
        xc.mappers.mapper_from_name('MapperPL15CMBK')


def test_maper_base_defaults():
    m = xc.mappers.MapperBase({'mask_name': 'mask',
                               'nside': 32})

    assert m.nside == 32
    assert m.mask_name == 'mask'
