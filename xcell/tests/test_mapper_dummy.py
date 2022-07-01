import numpy as np
import xcell as xc
import healpy as hp
import pyccl as ccl
import pytest


def get_config(dtype='galaxy_density', fsky=0.2):
    cosmo = {
      'Omega_c': 0.24,
      'Omega_b': 0.0493,
      'h': 0.72,
      'n_s': 0.9649,
      'sigma8': 0.78,
      'w0': -1,
      'wa': 0,
      'transfer_function': 'boltzmann_camb',
      'baryons_power_spectrum': 'nobaryons'}
    return {'seed': 0, 'nside': 32, 'fsky': fsky, 'cosmo': cosmo,
            'dtype': dtype, 'coords': 'C'}


def get_mapper(dtype='galaxy_density', fsky=0.2):
    return xc.mappers.MapperDummy(get_config(dtype, fsky))


def get_cl(dtype):
    config = get_config(dtype)
    cosmo_pars = config['cosmo']
    cosmo = ccl.Cosmology(**cosmo_pars)

    if config['dtype'] == 'generic':
        return np.ones(3*config['nside'])

    if config['dtype'] == 'galaxy_density':
        z, nz = np.loadtxt('xcell/tests/data/DESY1gc_dndz_bin0.txt',
                           usecols=(1, 3), unpack=True)
        b = np.ones_like(z)
        tracer = ccl.NumberCountsTracer(cosmo, dndz=(z, nz), bias=(z, b),
                                        has_rsd=None)
    elif config['dtype'] == 'galaxy_shear':
        z, nz = np.loadtxt('xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
                           usecols=(0, 1), unpack=True)
        tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    elif config['dtype'] == 'cmb_convergence':
        tracer = ccl.CMBLensingTracer(cosmo, z_source=1100)
    elif config['dtype'] == 'cmb_tSZ':
        tracer = ccl.tSZTracer(cosmo, z_max=3.)

    cl = ccl.angular_cl(cosmo, tracer, tracer, np.arange(3 * config['nside']))
    return cl


def test_smoke():
    get_mapper()


def test_get_mask():
    m = get_mapper(fsky=1)
    d = m.get_mask()
    assert np.all(d == 1)
    m = get_mapper(fsky=100)
    d = m.get_mask()
    assert np.all(d == 1)
    # TODO: Implement apodized mask
    # d = m.get_mask(fsky=0.5)


@pytest.mark.parametrize('dtyp', ['galaxy_density',
                                  'galaxy_shear',
                                  'cmb_convergence',
                                  'cmb_tSZ',
                                  'generic'])
def test_get_cl(dtyp):
    m = get_mapper(dtyp)
    cl_m = m.get_cl()

    cl = get_cl(dtyp)
    rdev = np.fabs(cl[cl != 0] / cl_m[cl_m != 0] - 1)
    assert np.max(rdev) < 1E-5


@pytest.mark.parametrize('dtyp', ['galaxy_density',
                                  'galaxy_shear',
                                  'cmb_convergence',
                                  'cmb_tSZ',
                                  'generic'])
def test_get_nz(dtyp):
    m = get_mapper(dtype=dtyp)
    nz = m.get_nz()
    if dtyp in ['cmb_convergence', 'cmb_tSZ', 'generic']:
        assert nz is None
    else:
        assert len(nz) == 2
        assert len(nz.shape) == 2


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    d0 = [hp.read_map('xcell/tests/data/dummy_signal_s0.fits')]

    # galaxy_density
    assert len(d) == 1
    assert np.std(d[0] - d0[0]) / np.std(d[0]) < 1e-2

    # galaxy_shear
    m = get_mapper('galaxy_shear')
    d = m.get_signal_map()
    assert len(d) == 2
    d0 = [hp.read_map('xcell/tests/data/dummy_signal_s2_0.fits'),
          hp.read_map('xcell/tests/data/dummy_signal_s2_1.fits')]
    assert np.std(d[0] - d0[0]) / np.std(d[0]) < 1e-2
    assert np.std(d[1] - d0[1]) / np.std(d[1]) < 1e-2


# Commented out because nl_copuled = 0 atm in MapperDummy
# def test_get_nl_coupled():
    # m = get_mapper()
    # Redicted value
    # nl_pred = 4*np.pi/m.npix
    # nl = m.get_nl_coupled()
    # assert np.all(np.fabs(nl-nl_pred) < 1E-5)


def test_get_dtype():
    m = get_mapper()
    assert 'galaxy_density' == m.get_dtype()


def test_get_spin():
    for dtype in ['galaxy_density', 'galaxy_shear',
                  'cmb_convergence', 'cmb_tSZ']:
        m = get_mapper(dtype)
        if dtype == 'galaxy_shear':
            assert m.get_spin() == 2
        else:
            assert m.get_spin() == 0


def test_cl_coupled():
    config = get_config()
    config['custom_auto'] = True
    config['custom_offset'] = np.log(2.)*0.001
    m = xc.mappers.MapperDummy(config)
    cl1 = m.get_cl_coupled()[0]

    mp = m.get_signal_map()
    msk = m.get_mask()
    cl2 = hp.anafast(mp*msk, iter=0)

    assert np.allclose(cl1-np.log(2.)*0.001, cl2,
                       atol=0, rtol=1E-10)


def test_cls_covar_coupled():
    offset = np.pi*0.002
    config = get_config()
    config['custom_auto'] = True
    config['custom_offset'] = offset
    m = xc.mappers.MapperDummy(config)
    cl1 = m.get_cls_covar_coupled()

    mp = m.get_signal_map()
    msk = m.get_mask()
    cl2 = hp.anafast(mp*msk, iter=0)

    assert np.allclose(cl1['cross'][0], cl2,
                       atol=0, rtol=1E-10)
    assert np.allclose(cl1['auto_11'][0]-offset, cl2,
                       atol=0, rtol=1E-10)
    assert np.allclose(cl1['auto_12'][0], cl2,
                       atol=0, rtol=1E-10)
    assert np.allclose(cl1['auto_11'][0]-offset, cl2,
                       atol=0, rtol=1E-10)


def test_get_beam():
    ell = np.arange(3*32)
    config = get_config()
    beam_infos = {'default': [],
                  'Gaussian': [{'type': 'Gaussian',
                               'FWHM_arcmin': 0.5}],
                  'PixWin': [{'type': 'PixWin',
                              'nside_native': config['nside']}],
                  'combined': [{'type': 'Gaussian',
                               'FWHM_arcmin': 0.5},
                               {'type': 'PixWin',
                                'nside_native': config['nside']}]}
    beam_outputs = {'default':
                    np.ones(3*32),
                    'Gaussian':
                    np.exp(-1.907e-09*ell*(ell+1)),
                    'PixWin':
                    hp.sphtfunc.pixwin(config['nside'])}
    beam_outputs['combined'] = beam_outputs['PixWin']
    beam_outputs['combined'] *= beam_outputs['Gaussian']
    for mode in beam_infos.keys():
        config['beam_info'] = beam_infos[mode]
        m = xc.mappers.MapperDummy(config)
        beamm = beam_outputs[mode]
        beam = m.get_beam()
        assert ((beam - beamm) < 1e-30).all
