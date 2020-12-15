import numpy as np
import xcell as xc
import healpy as hp


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog.fits'],
            'file_nz': 'xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
            'bin': '1', 'nside': 32, 'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperKV450(get_config())


def test_smoke():
    m = get_mapper()
    m.get_signal_map()
    m.get_mask()
    m.get_nmt_field()
    m.get_nl_coupled()
    print(np.ones(3))
    print(hp.nside2npix(m.nside))
