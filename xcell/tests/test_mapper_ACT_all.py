import xcell as xc
from pixell import enmap, reproject
import healpy as hp
import numpy as np
import os
import pytest


def get_config(wbeam=True):
    path = 'xcell/tests/data/'
    c = {'file_map': path+'act_zeros.fits.gz',
         'file_mask': path+'act_zeros.fits.gz',
         'file_noise': path+'act_zeros.fits.gz',
         'file_cross_noise': path+'act_zeros.fits.gz',
         'file_weights': path+'act_zeros.fits.gz',
         'file_beam': path+'act_zeros.fits.gz',
         'map_name': 'test',
         'mask_name': 'mask',
         'coords': 'C',
         'nside': 32}
    return c


@pytest.mark.parametrize('cls,spin', [(xc.mappers.MapperACTk, '0'),
                                      (xc.mappers.MapperACTtSZ, '0'),
                                      (xc.mappers.MapperACTCMB, '0')])
def test_get_spin(cls, spin):
    m = cls(get_config())
    assert m.get_spin() == int(spin)


@pytest.mark.parametrize('cls,typ', [(xc.mappers.MapperACTk,
                                      'cmb_convergence'),
                                     (xc.mappers.MapperACTtSZ,
                                      'cmb_tSZ'),
                                     (xc.mappers.MapperACTCMB,
                                      'cmb_kSZ')])
def test_get_dtype(cls, typ):
    m = cls(get_config())
    assert m.get_dtype() == typ


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTtSZ),
                                 (xc.mappers.MapperACTCMB),
                                 (xc.mappers.MapperACTk)])
def test_get_signal_map(cls):
    conf = get_config()
    conf['path_rerun'] = 'xcell/tests/data/'
    m = cls(conf)
    mb_pxll = enmap.read_map(conf['file_map'])
    mb = [reproject.healpix_from_enmap(mb_pxll,
                                       lmax=6000,
                                       nside=32)]
    mm = m.get_signal_map()[0]
    assert (len(mm)/12)**(1/2) == 32
    assert (mb == mm).all()
    fn = 'xcell/tests/data/ACT_test_signal_coordC_ns32.fits.gz'
    mrerun = hp.read_map(fn)
    assert (mrerun == mb).all()
    os.remove(fn)


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTtSZ),
                                 (xc.mappers.MapperACTCMB),
                                 (xc.mappers.MapperACTk)])
def test_get_mask(cls):
    conf = get_config()
    conf['path_rerun'] = 'xcell/tests/data/'
    m = cls(conf)
    mb_pxll = enmap.read_map(conf['file_mask'])
    mb = [reproject.healpix_from_enmap(mb_pxll,
                                       lmax=6000,
                                       nside=32)]
    mm = m.get_mask()
    assert (len(mm)/12)**(1/2) == 32
    assert (mb == mm).all()
    fn = 'xcell/tests/data/mask_mask_coordC_ns32.fits.gz'
    mrerun = hp.read_map(fn)
    assert (mrerun == mb).all()
    os.remove(fn)


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTtSZ),
                                 (xc.mappers.MapperACTCMB)])
def test_get_beam(cls):
    from scipy.interpolate import interp1d
    conf = get_config()
    conf['beam_info'] = [{'type': 'Custom',
                          'file': 'xcell/tests/data/custom_beam_act.txt'}]
    conf['path_rerun'] = 'xcell/tests/data/'
    m = cls(conf)
    beam = m.get_beam()
    beamm_file = np.loadtxt('xcell/tests/data/custom_beam_act.txt')
    beamm = np.transpose(beamm_file)[1]
    ells = np.transpose(beamm_file)[0]
    beamm_itp = interp1d(ells, np.log(beamm),
                         fill_value='extrapolate')
    beamm = np.exp(beamm_itp(np.arange(3*m.nside)))
    assert (beam-beamm < 0.0005).all()
