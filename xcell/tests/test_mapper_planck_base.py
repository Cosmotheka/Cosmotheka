import xcell as xc
import numpy as np
import pytest
import healpy as hp


def get_config():
    c = {'file_map': 'xcell/tests/data/map.fits',
         'file_hm1': 'xcell/tests/data/hm1_map.fits',
         'file_hm2': 'xcell/tests/data/hm2_map.fits',
         'file_mask': 'xcell/tests/data/map.fits',
         'file_gp_mask': 'xcell/tests/data/mask1.fits',
         'file_sp_mask': 'xcell/tests/data/mask2.fits',
         'gal_mask_mode': '0.2',
         'beam_fwhm_arcmin': 0.,
         'nside': 32}
    return c


@pytest.mark.parametrize('m', [xc.mappers.MapperP15tSZ(get_config()),
                               xc.mappers.MapperP18SMICA(get_config()),
                               xc.mappers.MapperP15CIB(get_config())])
def test_spin(m):
    assert m.get_spin() == 0


@pytest.mark.parametrize('m', [xc.mappers.MapperP15tSZ(get_config()),
                               xc.mappers.MapperP18SMICA(get_config()),
                               xc.mappers.MapperP15CIB(get_config())])
def test_get_signal_map(m):
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
    m1 = hp.read_map("xcell/tests/data/hm1_map.fits", verbose=False)
    m2 = hp.read_map("xcell/tests/data/hm2_map.fits", verbose=False)
    cl_cross_bm = hp.anafast(m1*mask, m2*mask, iter=0)
    nl_diff_bm = hp.anafast(0.5*(m1-m2)*mask, iter=0)
    # Typical C_ell value for comparison (~1E-3 in this case)
    cl_cross_scale = np.mean(np.fabs(cl_cross_bm[2:]))
    nl_diff_scale = np.mean(np.fabs(nl_diff_bm[2:]))
    assert np.allclose(cl_cross, cl_cross_bm,
                       rtol=0, atol=1E-10*cl_cross_scale)
    assert np.allclose(nl_diff, nl_diff_bm,
                       rtol=0, atol=1E-10*nl_diff_scale)
