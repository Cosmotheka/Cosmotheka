import shutil
import os
import numpy as np
from xcell.cls.cl import Cl
from xcell.cls.cov import Cov
import pymaster as nmt
from xcell.mappers import MapperDummy


# Remove previous test results
tmpdir1 = './xcell/tests/cls/dummy1'
if os.path.isdir(tmpdir1):
    shutil.rmtree(tmpdir1)
tmpdir2 = './xcell/tests/cls/dummy2'
if os.path.isdir(tmpdir2):
    shutil.rmtree(tmpdir2)


def get_config(fsky=0.2, fsky2=0.3):
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
        'transfer_function': 'eisenstein_hu',
        'baryons_power_spectrum': 'nobaryons',
    }
    dummy0 = {'mask_name': 'mask_dummy0', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky, 'seed': 0}
    dummy1 = {'mask_name': 'mask_dummy1', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky2, 'seed': 100}
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0, 'Dummy__1': dummy1},
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
            'output': tmpdir1}


def get_cl_class(fsky=0.2):
    data = get_config(fsky)
    return Cl(data, 'Dummy__0', 'Dummy__0')


def get_cov_class(fsky=0.2):
    data = get_config(fsky)
    return Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')


def test_smoke():
    get_cl_class()
    get_cov_class()


def test_cov_nlmarg():
    data = get_config(0.2)
    data['tracers']['Dummy__0']['nl_marginalize'] = True
    data['tracers']['Dummy__0']['nl_prior'] = 1E30
    data['tracers']['Dummy__0']['noise_level'] = 1E-5
    data['output'] = tmpdir2
    cov_class = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    cov = cov_class.get_covariance()
    num_l = len(cov)
    oo = np.ones(num_l)
    chi2 = np.dot(oo, np.linalg.solve(cov, oo))
    assert np.fabs(chi2) < 1E-5*num_l


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
    assert np.all(cl_class.wins == w.get_bandpower_windows())


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

    icov = np.linalg.inv(cov)
    icov_m = np.linalg.inv(cov_m)
    dCl = (cl_data - cl_m1)[0]

    chi2 = dCl.dot(icov).dot(dCl)
    chi2_m = dCl.dot(icov_m).dot(dCl)

    assert chi2/chi2_m - 1 < 1e-5


def test_cls_vs_namaster():
    # cls
    # Get cl from randomnly generated map ("data")
    cl_class = get_cl_class(fsky=1)
    ell, cl_data = cl_class.get_ell_cl()
    b = cl_class.get_NmtBin()

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()

    # NaMaster
    config = get_config()
    m = MapperDummy(config['tracers']['Dummy__0'])

    # True cl
    cl_m = m.get_cl()
    spin = m.get_spin()
    mask = m.get_mask()
    signal_map = m.get_signal_map()
    n_iter_sht = config['healpy']['n_iter_sht']
    n_iter_mcm = config['healpy']['n_iter_mcm']
    n_iter_cmcm = config['healpy']['n_iter_cmcm']
    # Compute Cl from map
    f = nmt.NmtField(mask, signal_map, spin=spin, n_iter=n_iter_sht)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(f, f, bins=b, n_iter=n_iter_mcm)
    cl_data_nmt_cp = nmt.compute_coupled_cell(f, f)
    cl_data_nmt = wsp.decouple_cell(cl_data_nmt_cp)
    rdev = cl_data / cl_data_nmt - 1
    assert np.max(np.abs(rdev)) < 1e-5
    # Couple true Cl
    cl_m_cp = wsp.couple_cell([cl_m])
    cl_m = wsp.decouple_cell(cl_m_cp)

    # Compute cov with NaMaster
    cwsp = nmt.NmtCovarianceWorkspace()
    cwsp.compute_coupling_coefficients(f, f, n_iter=n_iter_cmcm)
    cl_cov = cl_m_cp / np.mean(mask * mask)
    cov_nmt = nmt.gaussian_covariance(cwsp, spin, spin, spin, spin, cl_cov,
                                      cl_cov, cl_cov, cl_cov, wsp)

    icov = np.linalg.inv(cov)
    icov_nmt = np.linalg.inv(cov_nmt)
    dCl = (cl_data - cl_m)[0]

    chi2 = dCl.dot(icov).dot(dCl)
    chi2_m = dCl.dot(icov_nmt).dot(dCl)

    assert chi2/chi2_m - 1 < 1e-5


def test_symmetric():
    data = get_config()
    cl_class01 = Cl(data, 'Dummy__0', 'Dummy__1')
    cl_class10 = Cl(data, 'Dummy__1', 'Dummy__0')

    assert np.all(np.array(cl_class01.get_masks()) ==
                  np.array(cl_class10.get_masks()[::-1]))
    assert np.all(cl_class01.get_ell_cl()[1] == cl_class10.get_ell_cl()[1])
    assert np.all(cl_class01.get_ell_nl()[1] == cl_class10.get_ell_nl()[1])
    assert np.all(cl_class01.get_ell_nl_cp()[1] ==
                  cl_class10.get_ell_nl_cp()[1])
