import numpy as np
import xcell as xc
import healpy as hp


def get_config():
    return {'data_catalog': 'xcell/tests/data/catalog.fits',
            'file_mask': 'xcell/tests/data/map.fits',
            'file_nz': 'xcell/tests/data/2pt_NG_mcal_1110.fits',
            'coords': 'C', 'zbin': 2, 'nside': 32,
            'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperDESY1gc(get_config())


def test_smoke():
    get_mapper()


def test_get_binning():
    config = get_config()

    # No galaxies
    for b in [0, 4]:
        config['zbin'] = b
        m = xc.mappers.MapperDESY1gc(config)
        m.get_catalog()
        m._get_w()
        assert len(m.cat_data) == 0
        assert np.sum(np.fabs(m.w)) < 1E-1

    # All galaxies
    ngal = hp.nside2npix(config['nside'])
    config['zbin'] = 2
    m = xc.mappers.MapperDESY1gc(config)
    m.get_catalog()
    m._get_w()
    assert len(m.cat_data) == ngal
    # Weight = 2
    assert np.sum(np.fabs(m.w))-2*ngal < 1E-5


def test_get_mask():
    m = get_mapper()
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    # Redicted value
    nl_pred = 4*np.pi/m.npix
    nl = m.get_nl_coupled()
    assert np.all(np.fabs(nl-nl_pred) < 1E-5)


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz(dz=0)
    assert len(z) == len(nz) == 400
    z, nz = m.get_nz(dz=-0.3)
    assert len(z) == len(nz) == 370


def test_get_dtype():
    m = get_mapper()
    dtype = m.get_dtype()
    assert dtype == 'galaxy_density'
