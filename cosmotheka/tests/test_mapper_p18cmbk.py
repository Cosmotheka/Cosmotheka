import cosmotheka as xc
import numpy as np
import healpy as hp
import shutil
import os


def get_config():
    return {'file_klm': 'cosmotheka/tests/data/alm.fits',
            'file_mask': 'cosmotheka/tests/data/map.fits',
            'file_noise': 'cosmotheka/tests/data/nl.txt',
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
    klm = m._get_klm()
    alm_all, lmax = hp.read_alm(config['file_klm'], return_mmax=True)
    alm_all = m.rot.rotate_alm(alm_all)
    alm_all[0] = 0+0j
    fl = np.ones(lmax+1)
    fl[3*16:] = 0
    alm_cut = hp.almxfl(alm_all, fl, inplace=True)
    assert np.all(np.real(klm - alm_cut) == 0.)


def test_smoke():
    get_mapper()


def test_dtype():
    m = get_mapper()
    assert m.get_dtype() == 'cmb_convergence'


def test_spin():
    m = get_mapper()
    assert m.get_spin() == 0


def test_get_signal_map():
    config = get_config()
    config['path_rerun'] = 'cosmotheka/tests/data/'
    m = xc.mappers.MapperP18CMBK(config)
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d) < 0.02)

    fn = 'cosmotheka/tests/data/P18CMBK_signal_map_coordC_ns32.fits.gz'
    assert np.all(d == hp.read_map(fn))
    os.remove(fn)


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
    assert np.all(np.fabs(cl) < 1E-5)


def test_get_sims_fnames(tmp_path):
    rec_dir = tmp_path / 'rec_sims'
    in_dir = tmp_path / 'input_sims'
    rec_dir.mkdir()
    in_dir.mkdir()

    rec_ids = [2, 0, 1]
    in_ids = [1, 0, 2]
    rec_files = []
    in_files = []

    for i in rec_ids:
        f = rec_dir / f'sim_klm_{i:03d}.fits'
        f.write_bytes(b'')
        rec_files.append(str(f))

    for i in in_ids:
        f = in_dir / f'sky_klm_{i:03d}.fits'
        f.write_bytes(b'')
        in_files.append(str(f))

    config = get_config()
    config['sims_rec_path'] = str(rec_dir)
    config['sims_in_path'] = str(in_dir)
    mapper = xc.mappers.MapperP18CMBK(config)

    rec_sims, input_sims = mapper._get_sims_fnames()

    assert rec_sims == sorted(rec_files)
    assert input_sims == sorted(in_files)
