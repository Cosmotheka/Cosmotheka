import xcell as xc
import numpy as np
import healpy as hp


def get_config():
    return {'file_map': 'xcell/tests/data/map.fits',
            'file_hm1': None,
            'file_hm2': None,
            'file_mask': 'xcell/tests/data/map.fits',
            'file_gp_mask': None,
            'file_sp_mask': None,
            'gal_mask_mode': None,
            'nside': 32}


def get_mapper(c=None):
    if c is None:
        c = get_config()
    return xc.mappers.MapperP15tSZ(c)


def test_get_mask():
    m = get_mapper()
    npix = hp.nside2npix(m.nside)
    mask = m.get_mask()
    assert(sum(mask) == npix)


def test_get_nmt_field():
    import pymaster as nmt
    m = get_mapper()
    f = m.get_nmt_field()
    cl = nmt.compute_coupled_cell(f, f)[0]
    assert np.fabs(cl[0]-4*np.pi) < 1E-3
    assert np.all(np.fabs(cl[1:]) < 1E-5)
