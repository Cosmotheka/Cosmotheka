import cosmotheka as xc
import healpy as hp
import numpy as np
import os
import pytest


def get_config():
    path = 'cosmotheka/tests/data/'
    c = {'klm_file': path+'alm.fits',
         'file_mask': path+'map.fits',
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


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTDR6k)])
def test_get_signal_map(cls):
    conf = get_config()
    conf['path_rerun'] = 'cosmotheka/tests/data/'
    m = cls(conf)
    alms, mmax = hp.read_alm(conf['klm_file'], return_mmax=True)
    alms = alms.astype(np.complex128)
    alms = np.nan_to_num(alms)
    fl = np.ones(mmax+1)
    fl[3*m.nside:] = 0
    hp.almxfl(alms, fl, inplace=True)
    map = hp.alm2map(alms, nside=32)
    map = xc.mappers.utils.rotate_map(map, m._get_rotator('C'))

    m_pipe = m.get_signal_map()[0]
    assert (len(m_pipe)/12)**(1/2) == 32
    assert (m_pipe == map).all()

    path = 'cosmotheka/tests/data/'
    fn = path + 'ACT_DR6_kappa_test_baseline_signal_map_coordC_ns32.fits.gz'
    mr = hp.read_map(fn)
    assert (mr == map).all()
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
