# flake8: noqa
from .mapper_base import MapperBase
from .mapper_base import MapperSDSS
from .mapper_DESY1gc import MapperDESY1gc
from .mapper_DESY1wl import MapperDESY1wl
from .mapper_eBOSSQSO import MappereBOSSQSO
from .mapper_BOSSCMASS import MapperBOSSCMASS
from .mapper_BOSSLOWZ import MapperBOSSLOWZ
from .mapper_eBOSSELG import MappereBOSSELG
from .mapper_eBOSSLRG import MappereBOSSLRG
from .mapper_KV450 import MapperKV450
from .mapper_KiDS1000 import MapperKiDS1000
from .mapper_P18CMBK import MapperP18CMBK
from .mapper_DELS import MapperDELS
from .mapper_dummy import MapperDummy
from .utils import get_map_from_points


def mapper_from_name(name):
    subcs = MapperBase.__subclasses__()
    subcs.extend(MapperSDSS.__subclasses__())
    mappers = {m.__name__: m for m in subcs}
    if name in mappers:
        return mappers[name]
    else:
        raise ValueError(f"Unknown mapper {name}")
