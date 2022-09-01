# flake8: noqa
from .mapper_base import MapperBase
from .mapper_ACT_base import MapperACTBase
from .mapper_ACT_compsept import MapperACTCompSept
from .mapper_ACTk import MapperACTk
from .mapper_ACTtSZ import MapperACTtSZ
from .mapper_ACTCMB import MapperACTCMB
from .mapper_SDSS import MapperSDSS
from .mapper_Planck_base import MapperPlanckBase
from .mapper_DESY1gc import MapperDESY1gc
from .mapper_DESY1wl import MapperDESY1wl
from .mapper_eBOSS import MappereBOSS
from .mapper_BOSS import MapperBOSS
from .mapper_KV450 import MapperKV450
from .mapper_KiDS1000 import MapperKiDS1000
from .mapper_SPT import MapperSPT
from .mapper_P18CMBK import MapperP18CMBK
from .mapper_P15tSZ import MapperP15tSZ
from .mapper_P18SMICA import MapperP18SMICA
from .mapper_P15CIB import MapperP15CIB
from .mapper_CIBLenz import MapperCIBLenz
from .mapper_DELS import MapperDELS
from .mapper_2MPZ import Mapper2MPZ
from .mapper_WIxSC import MapperWIxSC
from .mapper_HSC_DR1wl import MapperHSCDR1wl
from .mapper_NVSS import MapperNVSS
from .mapper_CatWISE import MapperCatWISE
from .mapper_ROSAT import MapperROSATXray
from .mapper_dummy import MapperDummy
from .utils import (get_map_from_points, get_DIR_Nz,
                    get_rerun_data, save_rerun_data,
                    rotate_mask, rotate_map)

def mapper_from_name(name):
    def all_subclasses(cls):
        # Recursively find all subclasses (and their subclasses)
        # From https://stackoverflow.com/questions/3862310
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)])
    subcs = all_subclasses(MapperBase)
    mappers = {m.__name__: m for m in subcs}
    if name in mappers:
        return mappers[name]
    else:
        raise ValueError(f"Unknown mapper {name}")