import xcell as xc
import numpy as np
import healpy as hp
import shutil


def get_config():
    return {'file_klm': 'xcell/tests/data/alm.fits',
            'file_mask': 'xcell/tests/data/map.fits',
            'file_noise': 'xcell/tests/data/nl.txt',
            'mask_name': 'mask_CMBK',
            'mask_aposize': 3,  # Must be large than pixel size
            'mask_apotype': 'C1',
            'nside': 32, 'coords': 'C'}


def get_mapper():
    config = get_config()
    return xc.mappers.MapperP18CMBK(config)


def test_alm_cut():
    # Tests alm filtering for CMB kappa alms on low resolution pixels.
    config = get_config()
    config['nside'] = 16
    m = xc.mappers.MapperP18CMBK(config)
    m.get_signal_map()
    alm_all, lmax = hp.read_alm(config['file_klm'], return_mmax=True)
    alm_all = m.rot.rotate_alm(alm_all)
    fl = np.ones(lmax+1)
    fl[3*16:] = 0
    alm_cut = hp.almxfl(alm_all, fl, inplace=True)
    assert np.all(np.real(m.klm - alm_cut) == 0.)


def test_smoke():
    get_mapper()


def test_dtype():
    m = get_mapper()
    assert m.get_dtype() == 'cmb_convergence'


def test_spin():
    m = get_mapper()
    assert m.get_spin() == 0


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d-1) < 0.02)


def test_get_mask():
    c = get_config()
    c['path_rerun'] = './lite'
    m = xc.mappers.MapperP18CMBK(c)
    d = m.get_mask()
    assert np.all(np.fabs(d-1) < 1E-5)
    # Now read from lite path
    m2 = xc.mappers.MapperP18CMBK(c)
    d2 = m2.get_mask()
    assert np.all(np.fabs(d-d2) < 1E-10)
    shutil.rmtree('./lite')


def test_get_nl_coupled():
    m = get_mapper()
    nl = m.get_nl_coupled()
    cl = m.get_cl_fiducial()
    ell = m.get_ell()

    assert nl.shape == (1, 3*32)
    assert np.all(np.fabs(nl) < 1E-15)
    assert cl.shape == (1, 3*32)
    assert np.all(ell == np.arange(3 * 32))


def test_get_nmt_field():
    import pymaster as nmt
    m = get_mapper()
    f = m.get_nmt_field()
    cl = nmt.compute_coupled_cell(f, f)[0]
    assert np.fabs(cl[0]-4*np.pi) < 1E-3
    assert np.all(np.fabs(cl[1:]) < 1E-5)
