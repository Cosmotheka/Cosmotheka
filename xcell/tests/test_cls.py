import shutil
import os
import pymaster as nmt
import numpy as np
from xcell.cls.cl import Cl


# Remove previous test results
tmpdir = './xcell/tests/cls/dummy'
if os.path.isdir(tmpdir):
    shutil.rmtree(tmpdir)


def get_config():
    nside = 256
    dummy0 = get_config_dummy(nside=nside)
    dummy0.update({'mask_name': 'mask_dummy0',
                   'mapper_class': 'MapperDummy'})
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0},
            'cls': {'Dummy-Dummy': {'compute': 'all'}},
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


def get_config_dummy(l0=20., alpha=1., spin=0, nside=32):
    return {'nside': nside,
            'spin': spin,
            'l0': l0,
            'alpha': alpha
            }


def get_cl_class():
    data = get_config()
    return Cl(data, 'Dummy__0', 'Dummy__0')


def test_smoke():
    get_cl_class()


def test_get_ell_cl():
    # Get cl from map
    cl_class = get_cl_class()
    ell, cl = cl_class.get_ell_cl()

    # Get cl from mapper (the true one)
    m1, m2 = cl_class.get_mappers()
    w = cl_class.get_workspace()
    cl_m1 = m1.get_cl(np.arange(3 * m1.nside))
    cl_m1_cp = w.couple_cell([cl_m1])
    cl_m1 = w.decouple_cell(cl_m1_cp)

    # Compute covariance
    mask1, mask2 = cl_class.get_masks()
    cl_m1_cw = cl_m1_cp / np.mean(mask1 * mask2)
    cwsp = nmt.NmtCovarianceWorkspace()
    cwsp.compute_coupling_coefficients(*cl_class.get_nmt_fields())
    cov = nmt.covariance.gaussian_covariance(cwsp, 0, 0, 0, 0, cl_m1_cw,
                                             cl_m1_cw, cl_m1_cw, cl_m1_cw, w)

    # Check that true Cl is within 5sigma of data Cl
    sigma = np.sqrt(np.diag(cov))
    cl_p = cl + 5 * sigma
    cl_m = cl - 5 * sigma
    check = np.all((cl_m1[0] > cl_m[0]) * (cl_m1[0] < cl_p[0]))

    assert(check)
