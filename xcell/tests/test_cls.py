import numpy as np
import xcell as xc
from xcell.cls.cl import Cl
import healpy as hp
import shutil
import os


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
            'recompute': {'cls' : True,
                          'cov' : True,
                          'mcm' : True,
                          'cmcm' : True},
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
    cl_class = get_cl_class()
    ell, cl = cl_class.get_ell_cl()

    m1, m2 = cl_class.get_mappers()
    w = cl_class.get_workspace()
    cl_m1 = m1.get_cl(np.arange(3 * m1.nside))
    cl_m1 = w.decouple_cell(w.couple_cell([cl_m1]))

    from matplotlib import pyplot as plt
    plt.plot(ell, cl[0])
    plt.plot(ell, cl_m1[0])
    plt.savefig('prueba.png')
    plt.close()

    rdev = np.mean(cl/cl_m1 - 1)

    assert(np.max(np.abs(rdev)) < 0.05)
