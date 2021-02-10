import numpy as np
import xcell as xc
import healpy as hp
import pyccl as ccl


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
            'dtype': dtype}


def get_mapper(dtype='galaxy_density', fsky=0.2):
    return xc.mappers.MapperDummy(get_config(dtype, fsky))


def get_cl(dtype):
    config = get_config(dtype)
    cosmo_pars = config['cosmo']
    cosmo = ccl.Cosmology(**cosmo_pars)

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


def test_get_cl():
    for dtype in ['galaxy_density', 'galaxy_shear', 'cmb_convergence']:
        m = get_mapper(dtype)
        cl_m = m.get_cl()

        cl = get_cl(dtype)
        rdev = np.fabs(cl[cl != 0] / cl_m[cl_m != 0] - 1)
        assert np.max(rdev) < 1E-5


def test_get_signal_map():
    m = get_mapper()
    d = m.get_signal_map()
    d0 = [hp.read_map('xcell/tests/data/dummy_signal_s0.fits')]

    # galaxy_density
    assert len(d) == 1
    assert np.std(d[0] - d0[0]) / np.std(d[0]) < 1e-3
    # galaxy_shear
    m = get_mapper('galaxy_shear')
    d = m.get_signal_map()
    assert len(d) == 2
    d0 = [hp.read_map('xcell/tests/data/dummy_signal_s2_0.fits'),
          hp.read_map('xcell/tests/data/dummy_signal_s2_1.fits')]
    assert np.std(d[0] - d0[0]) / np.std(d[0]) < 1e-3
    assert np.std(d[1] - d0[1]) / np.std(d[1]) < 1e-3


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
    for dtype in ['galaxy_density', 'galaxy_shear', 'cmb_convergence']:
        m = get_mapper(dtype)
        if dtype in ['galaxy_density', 'cmb_convergence']:
            assert m.get_spin() == 0
        else:
            assert m.get_spin() == 2
