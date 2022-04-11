import xcell as xc
import numpy as np
import pytest
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d


def get_config(mode, wbeam=True):
    if mode == 'SMICA':
        c = {'file_map': 'xcell/tests/data/map.fits',
             'file_hm1': 'xcell/tests/data/hm1_map.fits',
             'file_hm2': 'xcell/tests/data/hm2_map.fits',
             'file_gp_mask': 'xcell/tests/data/mask1.fits',
             'gp_mask_mode': '0.2',
             'nside': 32}
    elif mode == 'tSZ':
        c = {'file_map': 'xcell/tests/data/map.fits',
             'file_hm1': 'xcell/tests/data/map.fits',
             'file_hm2': 'xcell/tests/data/map.fits',
             'file_mask': 'xcell/tests/data/map.fits',
             'file_gp_mask': 'xcell/tests/data/mask1.fits',
             'gp_mask_mode': '0.4',
             'nside': 32}
    elif mode == 'P15CIB':
        c = {'file_map': 'xcell/tests/data/map.fits',
             'file_hm1': 'xcell/tests/data/hm1_map.fits',
             'file_hm2': 'xcell/tests/data/hm2_map.fits',
             'file_mask': 'xcell/tests/data/map.fits',
             'file_gp_mask': 'xcell/tests/data/mask1.fits',
             'file_ps_mask': 'xcell/tests/data/mask2.fits',
             'gp_mask_mode': '0.2',
             'ps_mask_mode': ['100'],
             'nside': 32}
    elif mode == 'CIBLenz':
        c = {'file_map': 'xcell/tests/data/map.fits',
             'file_hm1': 'xcell/tests/data/hm1_map.fits',
             'file_hm2': 'xcell/tests/data/hm2_map.fits',
             'file_mask': 'xcell/tests/data/map.fits',
             'nside': 32}
    elif mode == 'SPT':
        c = {'file_map': 'xcell/tests/data/map.fits',
             'file_hm1': 'xcell/tests/data/hm1_map.fits',
             'file_hm2': 'xcell/tests/data/hm2_map.fits',
             'file_gp_mask': 'xcell/tests/data/mask1.fits',
             'file_ps_mask': 'xcell/tests/data/mask2.fits',
             'nside': 32, 'coords': 'C'}
    elif mode == 'base':
        c = {'file_map': 'xcell/tests/data/map.fits',
             'file_hm1': 'xcell/tests/data/hm1_map.fits',
             'file_hm2': 'xcell/tests/data/hm2_map.fits',
             'nside': 32}
    else:
        print('Mode not recognized')
    if wbeam:
        c['beam_info'] = [{'type': 'Gaussian',
                           'FWHM_arcmin': 0.5}]
    c['coords'] = c.get('coords', 'G')
    return c


def test_spin():
    m = xc.mappers.MapperPlanckBase(get_config('base'))
    assert m.get_spin() == 0


def test_get_signal_map():
    m = xc.mappers.MapperPlanckBase(get_config('base'))
    d = m.get_signal_map()
    assert len(d) == 1
    d = d[0]
    assert np.all(np.fabs(d-1) < 0.02)


@pytest.mark.parametrize('m',
                         [xc.mappers.MapperP15tSZ(get_config('tSZ')),
                          xc.mappers.MapperP18SMICA(get_config('SMICA')),
                          xc.mappers.MapperP15CIB(get_config('P15CIB')),
                          xc.mappers.MapperCIBLenz(get_config('CIBLenz')),
                          xc.mappers.MapperSPT(get_config('SPT'))])
def test_get_nl_coupled(m):
    nl = m.get_nl_coupled()
    assert np.mean(nl) < 0.001


@pytest.mark.parametrize('cls,mode', [(xc.mappers.MapperP15tSZ, 'tSZ'),
                                      (xc.mappers.MapperP18SMICA, 'SMICA'),
                                      (xc.mappers.MapperP15CIB, 'P15CIB'),
                                      (xc.mappers.MapperCIBLenz, 'CIBLenz'),
                                      (xc.mappers.MapperSPT, 'SPT')])
def test_get_cl_coupled(cls, mode):
    conf = get_config(mode)
    conf_ref = get_config('base')
    conf['file_map'] = 'xcell/tests/data/map_auto_test.fits'
    m = cls(conf)
    mask = m.get_mask()
    cl_cross = m.get_cl_coupled()[0]
    nl_diff = m.get_nl_coupled()[0]
    m1 = hp.read_map(conf_ref['file_hm1'])
    m2 = hp.read_map(conf_ref['file_hm2'])
    cl_cross_bm = hp.anafast(m1*mask, m2*mask, iter=0)
    nl_diff_bm = hp.anafast(0.5*(m1-m2)*mask, iter=0)
    # Typical C_ell value for comparison (~1E-3 in this case)
    cl_cross_scale = np.mean(np.fabs(cl_cross_bm[2:]))
    nl_diff_scale = np.mean(np.fabs(nl_diff_bm[2:]))
    assert np.allclose(cl_cross, cl_cross_bm,
                       rtol=0, atol=1E-10*cl_cross_scale)
    assert np.allclose(nl_diff, nl_diff_bm,
                       rtol=0, atol=1E-10*nl_diff_scale)


@pytest.mark.parametrize('cls,mode', [(xc.mappers.MapperP15tSZ, 'tSZ'),
                                      (xc.mappers.MapperP18SMICA, 'SMICA'),
                                      (xc.mappers.MapperP15CIB, 'P15CIB'),
                                      (xc.mappers.MapperCIBLenz, 'CIBLenz'),
                                      (xc.mappers.MapperSPT, 'SPT')])
def test_get_cls_covar_coupled(cls, mode):
    conf = get_config(mode)
    conf_ref = get_config('base')
    conf['file_map'] = 'xcell/tests/data/map_auto_test.fits'
    m = cls(conf)
    mask = m.get_mask()
    cls_cov = m.get_cls_covar_coupled()
    m1 = hp.read_map(conf_ref['file_hm1'])
    m2 = hp.read_map(conf_ref['file_hm2'])
    mc = hp.read_map(conf['file_map'])
    cls_bm = {'cross': hp.anafast(mc*mask, mc*mask, iter=0),
              'auto_11': hp.anafast(m1*mask, m1*mask, iter=0),
              'auto_12': hp.anafast(m1*mask, m2*mask, iter=0),
              'auto_22': hp.anafast(m2*mask, m2*mask, iter=0)}
    for k in cls_bm.keys():
        cl = cls_cov[k][0]
        cl_bm = cls_bm[k]
        scale = np.mean(np.fabs(cl[2:]))
        assert np.allclose(cl, cl_bm,
                           rtol=0, atol=1E-10*scale)


@pytest.mark.parametrize('cls,mode', [(xc.mappers.MapperP15tSZ, 'tSZ'),
                                      (xc.mappers.MapperP18SMICA, 'SMICA'),
                                      (xc.mappers.MapperP15CIB, 'P15CIB'),
                                      (xc.mappers.MapperCIBLenz, 'CIBLenz'),
                                      (xc.mappers.MapperSPT, 'SPT')])
def test_get_hm_maps(cls, mode):
    conf = get_config(mode)
    m = cls(conf)
    conf_ref = get_config('base')
    m1b = hp.read_map(conf_ref['file_hm1'])
    m2b = hp.read_map(conf_ref['file_hm2'])
    m1, m2 = m._get_hm_maps()
    assert np.all(m1 == m1b)
    assert np.all(m2 == m2b)


@pytest.mark.parametrize('cls,mode,frac', [(xc.mappers.MapperP15tSZ,
                                            'tSZ', 0.75),
                                           (xc.mappers.MapperP15CIB,
                                            'P15CIB', 0.5),
                                           (xc.mappers.MapperCIBLenz,
                                            'CIBLenz', 1),
                                           (xc.mappers.MapperP18SMICA,
                                            'SMICA', 0.75),
                                           (xc.mappers.MapperSPT,
                                            'SPT', 0.5)])
def test_get_mask(cls, mode, frac):
    m = cls(get_config(mode))
    npix = hp.nside2npix(m.nside)
    mask = m.get_mask()
    assert(np.fabs(sum(mask) - npix*frac) < 1E-5)


@pytest.mark.parametrize('cls,mode,typ', [(xc.mappers.MapperP15tSZ,
                                           'tSZ', 'cmb_tSZ'),
                                          (xc.mappers.MapperP15CIB,
                                           'P15CIB', 'generic'),
                                          (xc.mappers.MapperCIBLenz,
                                           'CIBLenz', 'generic'),
                                          (xc.mappers.MapperP18SMICA,
                                           'SMICA', 'cmb_temperature'),
                                          (xc.mappers.MapperSPT,
                                           'SPT', 'cmb_tSZ')])
def test_get_dtype(cls, mode, typ):
    m = cls(get_config(mode))
    assert m.get_dtype() == typ


@pytest.mark.parametrize('cls,mode,fwhm', [(xc.mappers.MapperP15tSZ,
                                            'tSZ', 10.),
                                           (xc.mappers.MapperP15CIB,
                                            'P15CIB', 5.),
                                           (xc.mappers.MapperCIBLenz,
                                            'CIBLenz', 5.),
                                           (xc.mappers.MapperP18SMICA,
                                            'SMICA', 5.)])
def test_get_fwhm(cls, mode, fwhm):
    m = cls(get_config(mode, wbeam=False))
    assert m.beam_info[0]['FWHM_arcmin'] == fwhm


def test_custom_beam():
    c = get_config('CIBLenz')
    c['beam_info'] = [{'type': 'Custom',
                       'file': 'xcell/tests/data/windowfunctions_test.csv',
                       'field': 'Wl_eff'}]

    m = xc.mappers.MapperCIBLenz(c)
    assert m.beam_info[0]['type'] == 'Custom'
    wf = m.get_beam()
    ell = np.arange(3*m.nside)
    windowfuncs = pd.read_csv('xcell/tests/data/windowfunctions_test.csv',
                              comment='#')
    pixwin = interp1d(np.array(windowfuncs['ell']),
                      np.log(np.array(windowfuncs['Wl_eff'])),
                      fill_value='extrapolate')
    wff = np.exp(pixwin(ell))
    assert np.all(wf == wff)
