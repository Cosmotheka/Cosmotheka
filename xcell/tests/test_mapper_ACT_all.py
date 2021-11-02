import xcell as xc
import numpy as np
import pytest
import healpy as hp


def get_config(wbeam=True):
    c = {'file_map': 'xcell/tests/data/act_dr4.01_s14s15_D56_lensing_mask.fits',
         'file_mask': 'xcell/tests/data/act_dr4.01_s14s15_D56_lensing_mask.fits',
         'file_noise': 'xcell/tests/data/act_dr4.01_s14s15_D56_lensing_mask.fits',
         'file_cross_noise': 'xcell/tests/data/act_dr4.01_s14s15_D56_lensing_mask.fits',
         'file_weights': 'xcell/tests/data/act_dr4.01_s14s15_D56_lensing_mask.fits',
         'file_beam': 'xcell/tests/data/mask2.fits',
         'nside': 32}
    if wbeam:
        c['beam_fwhm_arcmin'] = 0.
    return c
@pytest.mark.parametrize('cls,spin', [(xc.mappers.MapperACTtSZ,
                                      '0'),
                                     (xc.mappers.MapperACTk,
                                      '2')])
def test_get_spin(cls, spin):
    m = cls(get_config())
    assert m.get_spin() == spin


@pytest.mark.parametrize('cls,typ', [(xc.mappers.MapperACTtSZ,
                                      'cmb_tSZ'),
                                     (xc.mappers.MapperACTk,
                                      'weak_lensing')])
def test_get_dtype(cls, typ):
    m = cls(get_config())
    assert m.get_dtype() == typ


@pytest.mark.parametrize('cls,fwhm', [(xc.mappers.MapperACTtSZ, 10.),
                                      (xc.mappers.MapperACTk, 5.)])
def test_get_fwhm(cls, fwhm):
    m = cls(get_config(wbeam=False))
    assert m.beam_info == fwhm


def test_get_beam():
    # No beam
    m = xc.mappers.MapperACTBase(get_config())
    beam = m.get_beam()
    assert np.all(beam == 1.0)

    # 15-arcmin beam
    fwhm = 15.
    m = xc.mappers.MapperPlanckBase(get_config())
    m.beam_info = fwhm
    beam = m.get_beam()
    ls = np.arange(3*m.nside)
    bls = np.exp(-0.5*ls*(ls+1)*(fwhm*np.pi/180/60/2.355)**2)
    assert np.allclose(beam, bls, atol=0, rtol=1E-3)


def test_get_signal_map():
    m = xc.mappers.MapperACTBase(get_config())
    assert (len(m.get_signal_map()[0])/12)**(1/2)


def test_get_mask():
    m = xc.mappers.MapperACTBase(get_config())
    assert (len(m.get_mask())/12)**(1/2)


def test_get_noise():
    m = xc.mappers.MapperACTBase(get_config())
    assert (len(m.get_noise())/12)**(1/2)


def test_get_cross_noise():
    m = xc.mappers.MapperACTBase(get_config())
    assert (len(m.get_cross_noise())/12)**(1/2)


def test_get_weights():
    m = xc.mappers.MapperACTBase(get_config())
    assert (len(m.get_weights())/12)**(1/2)

