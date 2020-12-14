import xcell as xc
import pytest


def test_mapper_from_name():
    for nm in ['MapperP15CMBK', 'MappereBOSSQSO']:
        xc.mappers.mapper_from_name(nm)

    with pytest.raises(ValueError):
        xc.mappers.mapper_from_name('MapperPL15CMBK')
