import xcell as xc
import numpy as np
import healpy as hp


def get_config():
    return {'file_map': 'xcell/tests/data/map.fits',
            'file_hm1': 'xcell/tests/data/hm1_map.fits',
            'file_hm2': 'xcell/tests/data/hm2_map.fits',
            'file_mask': 'xcell/tests/data/map.fits',
            'file_gp_mask': 'xcell/tests/data/mask1.fits',
            'file_sp_mask': 'xcell/tests/data/mask2.fits',
            'gal_mask_mode': '20%',
            'nside': 32}


def get_mappers(c=None):
    if c is None:
        c = get_config()
    mappers = [xc.mappers.MapperP18tSZ(c),
               xc.mappers.MapperP18SMICA_NOSZ(c),
               xc.mappers.MapperP15CIB(c)]
    return mappers


def test_smoke():
    get_mappers()


def test_spin():
    mappers = get_mappers()
    for m in mappers:
        assert m.get_spin() == 0


def test_get_signal_map():
    mappers = get_mappers()
    for m in mappers:
        d = m.get_signal_map()
        assert len(d) == 1
        d = d[0]
        assert np.all(np.fabs(d-1) < 0.02)


def test_get_mask():
    # Checks whether the intersection of the masks
    # is done properly
    mappers = get_mappers()
    for m in mappers:
        npix = hp.nside2npix(m.nside)
        mask = m.get_mask()
        assert(sum(mask) == npix/2)


def test_get_nmt_field():
    import pymaster as nmt
    c = get_config()
    c['file_gp_mask'] = None
    c['file_sp_mask'] = None
    mappers = get_mappers(c)
    for m in mappers:
        f = m.get_nmt_field()
        cl = nmt.compute_coupled_cell(f, f)[0]
        # assert np.fabs(cl[0]-np.pi) < 1E-3
        # drop the 4 to account for smaller mask
        assert np.fabs(cl[0]-4*np.pi) < 1E-3
        assert np.all(np.fabs(cl[1:]) < 1E-5)


def test_get_nl_coupled():
    mappers = get_mappers()
    for m in mappers:
        nl = m.get_nl_coupled()
        assert np.mean(nl) < 0.001


def test_get_cl_coupled():
    c = get_config()
    c['file_map'] = 'xcell/tests/data/hm1_map.fits'
    mappers = get_mappers(c)
    for m in mappers:
        nl = m.get_nl_coupled()
        cl = m.get_cl_coupled(mode='Auto')
        cl_signal = m.get_cl_coupled(mode='Cross')

    assert nl.shape == (1, 3*32)
    assert cl.shape == (1, 3*32)
    assert cl_signal.shape == (1, 3*32)
    assert abs(np.mean((cl[0]-nl[0]-cl_signal[0])[20:])) < 0.0005
