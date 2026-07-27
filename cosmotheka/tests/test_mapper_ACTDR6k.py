import cosmotheka as xc
import healpy as hp
import numpy as np
import os
import pytest


def get_config():
    path = 'cosmotheka/tests/data/'
    c = {'klm_file': path+'alm.fits',
         'file_mask': path+'map.fits',
         'file_noise': path+'nl_dr6k.txt',
         'map_name': 'DR6_kappa_test',
         'mask_name': 'DR6_kappa_test',
         'coords': 'C',
         'nside': 32,
         'mask_threshold': 0.1,
         'variant': 'baseline'}
    return c


@pytest.mark.parametrize('cls,spin', [(xc.mappers.MapperACTDR6k, '0')])
def test_get_spin(cls, spin):
    m = cls(get_config())
    assert m.get_spin() == int(spin)


@pytest.mark.parametrize('cls,typ', [(xc.mappers.MapperACTDR6k,
                                      'cmb_convergence')])
def test_get_dtype(cls, typ):
    m = cls(get_config())
    assert m.get_dtype() == typ


def test_get_nl_coupled():
    conf = get_config()
    m = xc.mappers.MapperACTDR6k(conf)
    nl = m.get_nl_coupled()
    msk = m.get_mask()
    w2 = np.mean(msk**2)
    assert np.allclose(nl, w2)


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTDR6k)])
def test_get_signal_map(cls):
    # Same test as in test_mapper_P18CMBK.py, but for ACTDR6k mapper.
    config = get_config()
    config['path_rerun'] = 'cosmotheka/tests/data/'
    m = cls(config)
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d) < 0.02)

    path = 'cosmotheka/tests/data/'
    fn = path + 'ACT_DR6_kappa_test_baseline_signal_map_coordC_ns32.fits.gz'
    assert np.all(d == hp.read_map(fn))
    os.remove(fn)


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTDR6k)])
def test_get_mask(cls):
    conf = get_config()
    conf['path_rerun'] = 'cosmotheka/tests/data/'
    m = cls(conf)

    mask = hp.read_map(conf['file_mask'])
    mask = hp.ud_grade(mask, 32)
    mask = xc.mappers.utils.rotate_mask(mask, m._get_rotator('C'))
    mask[~(mask > conf["mask_threshold"])] = 0

    mask_pipe = m.get_mask()
    assert (len(mask_pipe)/12)**(1/2) == 32
    assert (mask_pipe == mask).all()

    path = 'cosmotheka/tests/data/'
    fn = path + 'mask_DR6_kappa_test_baseline_coordC_ns32.fits.gz'
    mr = hp.read_map(fn)
    assert (mr == mask).all()
    os.remove(fn)


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
        f = rec_dir / f'kappa_alm_sim_act_dr6_lensing_v1_baseline_{i:03d}.fits'
        f.write_bytes(b'')
        rec_files.append(str(f))

    for i in in_ids:
        f = in_dir / f'input_kappa_alm_sim_{i:03d}.fits'
        f.write_bytes(b'')
        in_files.append(str(f))

    config = get_config()
    config['sims_rec_path'] = str(rec_dir)
    config['sims_in_path'] = str(in_dir)
    mapper = xc.mappers.MapperACTDR6k(config)

    rec_sims, input_sims = mapper._get_sims_fnames()

    assert rec_sims == sorted(rec_files)
    assert input_sims == sorted(in_files)

    # Check that it raises an error if the number of sims is different
    os.remove(rec_files[0])
    with pytest.raises(
        ValueError, match="Number of reconstructed and input sims"
    ):
        mapper._get_sims_fnames()
