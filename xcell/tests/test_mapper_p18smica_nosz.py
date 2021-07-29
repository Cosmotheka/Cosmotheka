import xcell as xc
import numpy as np
import healpy as hp


def get_config():
    return {'file_map': 'xcell/tests/data/map.fits',
            'file_hm1': 'xcell/tests/data/hm1_map.fits',
            'file_hm2': 'xcell/tests/data/hm2_map.fits',
            'file_mask': None,
            'file_gp_mask': 'xcell/tests/data/mask1.fits',
            'file_sp_mask': 'xcell/tests/data/mask2.fits',
            'gal_mask_mode': '0.2',
            'nside': 32}


def get_mapper(c=None):
    if c is None:
        c = get_config()
    return xc.mappers.MapperP18SMICA_NOSZ(c)


def test_get_mask():
    m = get_mapper()
    npix = hp.nside2npix(m.nside)
    mask = m.get_mask()
    assert(sum(mask) == npix/2)


def test_get_nmt_field():
    import pymaster as nmt
    m = get_mapper()
    f = m.get_nmt_field()
    cl = nmt.compute_coupled_cell(f, f)[0]
    assert np.fabs(cl[0]-np.pi) < 1E-3
    assert np.mean(np.fabs(cl[:-20]) < 1E-30)
