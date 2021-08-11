import xcell as xc
import numpy as np
import pytest
import healpy as hp


def get_config(wbeam=True):
    c = {'file_map': 'xcell/tests/data/map.fits',
         'file_hm1': 'xcell/tests/data/hm1_map.fits',
         'file_hm2': 'xcell/tests/data/hm2_map.fits',
         'file_mask': 'xcell/tests/data/map.fits',
         'file_gp_mask': 'xcell/tests/data/mask1.fits',
         'file_sp_mask': 'xcell/tests/data/mask2.fits',
         'gal_mask_mode': '0.2',
         'nside': 32}
    if wbeam:
        c['beam_fwhm_arcmin'] = 0.
    return c


def test_spin():
    m = xc.mappers.MapperPlanckBase(get_config())
    assert m.get_spin() == 0


def test_get_signal_map():
    m = xc.mappers.MapperPlanckBase(get_config())
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d-1) < 0.02)


@pytest.mark.parametrize('m', [xc.mappers.MapperP15tSZ(get_config()),
                               xc.mappers.MapperP18SMICA(get_config()),
                               xc.mappers.MapperP15CIB(get_config())])
def test_get_nl_coupled(m):
    nl = m.get_nl_coupled()
    assert np.mean(nl) < 0.001


def test_get_beam():
    # No beam
    m = xc.mappers.MapperPlanckBase(get_config())
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


@pytest.mark.parametrize('cls', [xc.mappers.MapperP15tSZ,
                                 xc.mappers.MapperP18SMICA,
                                 xc.mappers.MapperP15CIB])
def test_get_cl_coupled(cls):
    conf = get_config()
    conf['file_map'] = 'xcell/tests/data/map_auto_test.fits'
    m = cls(conf)
    mask = m.get_mask()
    cl_cross = m.get_cl_coupled()[0]
    nl_diff = m.get_nl_coupled()[0]
    m1 = hp.read_map(conf['file_hm1'], verbose=False)
    m2 = hp.read_map(conf['file_hm2'], verbose=False)
    cl_cross_bm = hp.anafast(m1*mask, m2*mask, iter=0)
    nl_diff_bm = hp.anafast(0.5*(m1-m2)*mask, iter=0)
    # Typical C_ell value for comparison (~1E-3 in this case)
    cl_cross_scale = np.mean(np.fabs(cl_cross_bm[2:]))
    nl_diff_scale = np.mean(np.fabs(nl_diff_bm[2:]))
    assert np.allclose(cl_cross, cl_cross_bm,
                       rtol=0, atol=1E-10*cl_cross_scale)
    assert np.allclose(nl_diff, nl_diff_bm,
                       rtol=0, atol=1E-10*nl_diff_scale)


@pytest.mark.parametrize('cls,frac', [(xc.mappers.MapperP15tSZ, 1),
                                      (xc.mappers.MapperP15CIB, 1),
                                      (xc.mappers.MapperP18SMICA, 0.5)])
def test_get_mask(cls, frac):
    m = cls(get_config())
    npix = hp.nside2npix(m.nside)
    mask = m.get_mask()
    assert(np.fabs(sum(mask) - npix*frac) < 1E-5)


@pytest.mark.parametrize('cls,typ', [(xc.mappers.MapperP15tSZ,
                                      'cmb_tSZ'),
                                     (xc.mappers.MapperP15CIB,
                                      'generic'),
                                     (xc.mappers.MapperP18SMICA,
                                      'cmb_temperature')])
def test_get_dtype(cls, typ):
    m = cls(get_config())
    assert m.get_dtype() == typ


@pytest.mark.parametrize('cls,fwhm', [(xc.mappers.MapperP15tSZ, 10.),
                                      (xc.mappers.MapperP15CIB, 5.),
                                      (xc.mappers.MapperP18SMICA, 5.)])
def test_get_fwhm(cls, fwhm):
    m = cls(get_config(wbeam=False))
    assert m.beam_info == fwhm
