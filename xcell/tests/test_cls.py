import shutil
import os
import numpy as np
from xcell.cls.cl import Cl
from xcell.cls.cov import Cov


# Remove previous test results
tmpdir = './xcell/tests/cls/dummy'
if os.path.isdir(tmpdir):
    shutil.rmtree(tmpdir)


def get_config(fsky=0.2):
    nside = 32
    # Set only the necessary entries. Leave the others to their default value.
    cosmo = {
        # Planck 2018: Table 2 of 1807.06209
        # Omega_m: 0.3133
        'Omega_c': 0.2640,
        'Omega_b': 0.0493,
        'h': 0.6736,
        'n_s': 0.9649,
        'sigma8': 0.8111,
        'w0': -1,
        'wa': 0,
        'transfer_function': 'boltzmann_camb',
        'baryons_power_spectrum': 'nobaryons',
    }
    dummy0 = {'mask_name': 'mask_dummy0', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky}
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0},
            'cls': {'Dummy-Dummy': {'compute': 'all'}},
            'cov': {'fiducial': {'cosmo': cosmo, 'gc_bias':  False,
                                 'wl_m': False, 'wl_ia': False}},
            'bpw_edges': bpw_edges,
            'healpy': {'n_iter_sht': 0,
                       'n_iter_mcm': 3,
                       'n_iter_cmcm': 3,
                       'nside': nside},
            'recompute': {'cls': True,
                          'cov': True,
                          'mcm': True,
                          'cmcm': True},
            'output': tmpdir}


def get_cl_class(fsky=0.2):
    data = get_config(fsky)
    return Cl(data, 'Dummy__0', 'Dummy__0')


def get_cov_class(fsky=0.2):
    data = get_config(fsky)
    return Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')


def test_smoke():
    get_cl_class()
    get_cov_class()


def test_get_ell_cl():
    # Get cl from map
    cl_class = get_cl_class()
    ell, cl = cl_class.get_ell_cl()

    # Get cl from mapper (the true one)
    m1, m2 = cl_class.get_mappers()
    w = cl_class.get_workspace()
    cl_m1 = m1.get_cl()
    cl_m1_cp = w.couple_cell([cl_m1])
    cl_m1 = w.decouple_cell(cl_m1_cp)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()

    # Check that true Cl is within 5sigma of data Cl
    sigma = np.sqrt(np.diag(cov))

    assert np.all(np.fabs(cl_m1 - cl) < 5 * sigma)


def test_get_covariance():
    # Get cl from randomnly generated map ("data")
    cl_class = get_cl_class(fsky=1)
    ell, cl_data = cl_class.get_ell_cl()

    # Get cl from mapper (the true one)
    m1, m2 = cl_class.get_mappers()
    w = cl_class.get_workspace()
    cl_m1 = m1.get_cl()
    cl_m1_cp = w.couple_cell([cl_m1])
    cl_m1 = w.decouple_cell(cl_m1_cp)

    # # With no mask, there should not be any coupling
    # rdev = cl_m1_cp / cl_m1 - 1
    # assert np.max(np.abs(rdev) < 1e-5)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()

    cov_m = np.zeros_like(cov)
    diag = (2 * cl_m1 ** 2) / (2 * ell + 1)
    np.fill_diagonal(cov_m, diag)
    #

    icov = np.linalg.inv(cov)
    icov_m = np.linalg.inv(cov_m)
    dCl = (cl_data - cl_m1)[0]

    chi2 = dCl.dot(icov).dot(dCl)
    chi2_m = dCl.dot(icov_m).dot(dCl)

    assert chi2/chi2_m - 1 < 1e-5
