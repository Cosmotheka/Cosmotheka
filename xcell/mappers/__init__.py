from .mapper_base import MapperBase
from .mapper_DESY1gc import MapperDESY1gc
from .mapper_DESY1wl import MapperDESY1wl
from .mapper_eBOSSQSO import MappereBOSSQSO
from .mapper_KV450 import MapperKV450
from .mapper_P15CMBK import MapperP15CMBK


def mapper_from_name(name):
    mappers = {m.__name__: m for m in MapperBase.__subclasses__()}
    if name in mappers:
        return mappers[name]
    else:
        return ValueError(f"Unknown mapper {name}")
