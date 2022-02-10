import xcell as xc
from pixell import enmap, reproject
import healpy as hp
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
         'nside': 32}
    return c


@pytest.mark.parametrize('cls,spin', [(xc.mappers.MapperACTk, '0'),
                                      (xc.mappers.MapperACTtSZ, '0')])

def test_get_spin(cls, spin):
    m = cls(get_config())
    assert m.get_spin() == int(spin)

@pytest.mark.parametrize('cls,typ', [(xc.mappers.MapperACTk,
                                      'cmb_convergence'),
                                     (xc.mappers.MapperACTtSZ,
                                      'cmb_tSZ')])
def test_get_dtype(cls, typ):
    m = cls(get_config())
    assert m.get_dtype() == typ


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTtSZ),
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
    fn = 'xcell/tests/data/ACT_test_signal.fits.gz'
    mrerun = hp.read_map(fn)
    assert (mrerun == mb).all()
    os.remove(fn)


def test_get_mask():
    conf = get_config()
    conf['path_rerun'] = 'xcell/tests/data/'
    m = xc.mappers.MapperACTBase(conf)
    mb_pxll = enmap.read_map(conf['file_mask'])
    mb = [reproject.healpix_from_enmap(mb_pxll,
                                       lmax=6000,
                                       nside=32)]
    mm = m.get_mask()
    assert (len(mm)/12)**(1/2) == 32
    assert (mb == mm).all()
    fn = 'xcell/tests/data/ACT_test_mask.fits.gz'
    mrerun = hp.read_map(fn)
    assert (mrerun == mb).all()
    os.remove(fn)
