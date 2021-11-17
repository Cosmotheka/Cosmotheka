import xcell as xc
from pixell import enmap, reproject
import pytest


def get_config(wbeam=True):
    path = 'xcell/tests/data/'
    c = {'file_map':
         path+'test_pixell_mask.fits',
         'file_mask':
         path+'test_pixell_mask.fits',
         'file_noise':
         path+'test_pixell_mask.fits',
         'file_cross_noise':
         path+'test_pixell_mask.fits',
         'file_weights':
         path+'test_pixell_mask.fits',
         'file_beam':
         path+'test_pixell_mask.fits',
         'nside': 32}
    return c


@pytest.mark.parametrize('cls,spin', [(xc.mappers.MapperACTk,
                                      '0')])
def test_get_spin(cls, spin):
    m = cls(get_config())
    assert m.get_spin() == int(spin)


@pytest.mark.parametrize('cls,typ', [(xc.mappers.MapperACTk,
                                      'cmb_convergence')])
def test_get_dtype(cls, typ):
    m = cls(get_config())
    assert m.get_dtype() == typ


@pytest.mark.parametrize('cls', [(xc.mappers.MapperACTk)])
def test_get_signal_map(cls):
    conf = get_config()
    m = cls(get_config())
    mb_pxll = enmap.read_map(conf['file_map'])
    mb = [reproject.healpix_from_enmap(mb_pxll,
                                       lmax=6000,
                                       nside=32)]
    assert (len(m.get_signal_map()[0])/12)**(1/2) == 32
    assert (mb == m.get_signal_map()).all


def test_get_mask():
    conf = get_config()
    m = xc.mappers.MapperACTBase(conf)
    mb_pxll = enmap.read_map(conf['file_mask'])
    mb = [reproject.healpix_from_enmap(mb_pxll,
                                       lmax=6000,
                                       nside=32)]
    assert (len(m.get_mask())/12)**(1/2) == 32
    assert (mb == m.get_mask()).all
