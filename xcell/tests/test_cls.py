import shutil
import os
import numpy as np
from xcell.cls.cl import Cl, ClFid
from xcell.cls.cov import Cov
import pymaster as nmt
from xcell.mappers import MapperDummy
import pytest
import pyccl as ccl


# Remove previous test results
tmpdir1 = './xcell/tests/cls/dummy1'
if os.path.isdir(tmpdir1):
    shutil.rmtree(tmpdir1)
tmpdir2 = './xcell/tests/cls/dummy2'
if os.path.isdir(tmpdir2):
    shutil.rmtree(tmpdir2)


def get_config(fsky=0.2, fsky2=0.3,
               dtype0='galaxy_density',
               dtype1='galaxy_density'):
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
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky, 'seed': 0,
              'dtype': dtype0}
    dummy1 = {'mask_name': 'mask_dummy1', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky2, 'seed': 100,
              'dtype': dtype1}
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
    shutil.rmtree(tmpdir1)


def test_get_nmtbin():
    # This test checks that the bpw_edges can be read from the
    # global part of the yaml file or from the context of one
    # of the cross-correlations.
    # 1. From global
    data = get_config()
    cl1 = Cl(data, 'Dummy__0', 'Dummy__0')
    shutil.rmtree(tmpdir1)
    # 2. From cross-correlations
    data['cls']['Dummy-Dummy']['bpw_edges'] = data.pop('bpw_edges')
    cl2 = Cl(data, 'Dummy__0', 'Dummy__0')
    shutil.rmtree(tmpdir1)
    # Check they are the same
    b1 = cl1.get_NmtBin()
    b2 = cl2.get_NmtBin()
    assert np.all(b1.get_effective_ells() == b2.get_effective_ells())


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
    shutil.rmtree(tmpdir2)


def test_file_inconsistent_errors():
    clo = get_cl_class()
    ell, cl = clo.get_ell_cl()
    # Change bpws and try to read file
    data = get_config(0.2)
    data['bpw_edges'] = data['bpw_edges'][:-1]
    data['recompute']['cls'] = False
    data['recompute']['mcm'] = False
    os.remove(os.path.join(tmpdir1, 'data.yml'))
    clo2 = Cl(data, 'Dummy__0', 'Dummy__0')
    with pytest.raises(ValueError):
        clo2.get_ell_cl()
    shutil.rmtree(tmpdir1)


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
    shutil.rmtree(tmpdir1)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()
    shutil.rmtree(tmpdir1)

    # Check that true Cl is within 5sigma of data Cl
    sigma = np.sqrt(np.diag(cov))

    assert np.all(np.fabs(cl_m1 - cl) < 5 * sigma)
    assert np.all(cl_class.wins == w.get_bandpower_windows())


def test_get_ell_cl_cp():
    # Get cl from map
    cl_class = get_cl_class()
    ell, cl = cl_class.get_ell_cl()
    ell, cl_cp = cl_class.get_ell_cl_cp()

    w = cl_class.get_workspace()
    cl2 = w.decouple_cell(cl_cp)
    assert np.all(np.fabs(cl / cl2 - 1) < 1e-10)


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
    diag = (2 * cl_m1 ** 2) / (2 * ell + 1) / 4
    np.fill_diagonal(cov_m, diag)

    icov = np.linalg.inv(cov)
    icov_m = np.linalg.inv(cov_m)
    dCl = (cl_data - cl_m1)[0]

    chi2 = dCl.dot(icov).dot(dCl)
    chi2_m = dCl.dot(icov_m).dot(dCl)

    assert np.fabs(chi2/chi2_m) - 1 < 0.01
    shutil.rmtree(tmpdir1)


def test_cls_vs_namaster():
    # cls
    # Get cl from randomnly generated map ("data")
    cl_class = get_cl_class()
    ell, cl_data = cl_class.get_ell_cl()
    b = cl_class.get_NmtBin()
    shutil.rmtree(tmpdir1)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()
    shutil.rmtree(tmpdir1)

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
    assert np.fabs(chi2/chi2_m) - 1 < 1e-5

    # Compute bandpower windows
    win = cl_class.get_bandpower_windows()
    bpwin = wsp.get_bandpower_windows()
    assert np.all(win == bpwin)


def test_symmetric():
    data = get_config()
    # Request only 'auto' to test if read_symmetric works in the case you have
    # 'auto but you need the cross for the covariance
    data['cls']['Dummy-Dummy']['compute'] = 'auto'
    cl_class01 = Cl(data, 'Dummy__0', 'Dummy__1')
    os.remove(os.path.join(tmpdir1, 'data.yml'))
    cl_class10 = Cl(data, 'Dummy__1', 'Dummy__0')

    fname = os.path.join(cl_class10.outdir, 'cl_Dummy__1_Dummy__0.npz')
    assert not os.path.isfile(fname)
    fname = os.path.join(cl_class10.outdir, 'w__mask_dummy1_mask_dummy0.fits')
    assert not os.path.isfile(fname)
    assert np.all(np.array(cl_class01.get_masks()) ==
                  np.array(cl_class10.get_masks()[::-1]))
    assert np.all(cl_class01.get_ell_cl()[1] == cl_class10.get_ell_cl()[1])
    assert np.all(cl_class01.get_ell_nl()[1] == cl_class10.get_ell_nl()[1])
    assert np.all(cl_class01.get_ell_nl_cp()[1] ==
                  cl_class10.get_ell_nl_cp()[1])
    shutil.rmtree(tmpdir1)


def test_symmetric_fid():
    data = get_config()
    # Request only 'auto' to test if read_symmetric works in the case you have
    # 'auto but you need the cross for the covariance
    data['cls']['Dummy-Dummy']['compute'] = 'auto'
    cl_class01 = ClFid(data, 'Dummy__0', 'Dummy__1')
    os.remove(os.path.join(tmpdir1, 'data.yml'))
    cl_class10 = ClFid(data, 'Dummy__1', 'Dummy__0')

    fname = os.path.join(cl_class10.outdir, 'cl_Dummy__1_Dummy__0.npz')
    assert not os.path.isfile(fname)
    assert np.all(cl_class01.get_ell_cl()[1] == cl_class10.get_ell_cl()[1])
    shutil.rmtree(tmpdir1)


def test_cov_nonoverlap():
    data = get_config(fsky=0.2, fsky2=0.2)
    data['tracers']['Dummy__0']['dec0'] = 0.
    data['tracers']['Dummy__1']['dec0'] = 180.
    covc = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__1', 'Dummy__1')
    cov = covc.get_covariance()
    shutil.rmtree(tmpdir1)
    assert np.all(cov == 0)


def test_cov_mmarg():
    sm = 0.1
    data = get_config(dtype0='galaxy_shear')
    data['tracers']['Dummy__0']['sigma_m'] = sm

    # Homemade marginalized covariance
    # First, get decoupled power spectra
    # Theory power spectra
    clf = ClFid(data, 'Dummy__0', 'Dummy__0')
    _, cl = clf.get_ell_cl()
    shutil.rmtree(tmpdir1)
    # Binning
    clc = Cl(data, 'Dummy__0', 'Dummy__0')
    wp = clc.get_bandpower_windows()
    shutil.rmtree(tmpdir1)
    ncl, nbpw, _, nl = wp.shape
    wp = wp.reshape((ncl*nbpw, ncl*nl))
    cl = cl.reshape(ncl*nl)
    cl = np.dot(wp, cl)
    # Marginalized covariance term
    covmargb = 4*sm**2*cl[:, None]*cl[None, :]

    # Do with xCell
    covc = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    covmarg = covc.get_covariance_m_marg()
    shutil.rmtree(tmpdir1)
    assert np.amax(np.fabs(covmarg-covmargb))/np.mean(covmarg) < 1E-5


@pytest.mark.parametrize('perm', [
    [0, 0, 0, 1],  # 00, 02
    [0, 0, 1, 1],  # 00, 22
    [0, 1, 0, 0],  # 02, 00
    [0, 1, 0, 1],  # 02, 02
    [0, 1, 1, 1],  # 02, 22
    [1, 1, 0, 0],  # 22, 00
    [1, 1, 0, 1],  # 22, 02
    [1, 1, 1, 1]])  # 22, 22
def test_cov_spin0(perm):
    nmp = [1, 2]
    nmaps = [nmp[p] for p in perm]
    ncls1 = nmaps[0]*nmaps[1]
    ncls2 = nmaps[2]*nmaps[3]
    data = get_config(fsky=1., fsky2=1.,
                      dtype0='galaxy_density',
                      dtype1='galaxy_shear')
    data['tracers']['Dummy__0']['noise_level'] = 1E-5
    data['tracers']['Dummy__1']['noise_level'] = 1E-10
    nbpw = len(data['bpw_edges'])

    # Spin-2 covariance class
    covc2 = Cov(data,
                f'Dummy__{perm[0]}',
                f'Dummy__{perm[1]}',
                f'Dummy__{perm[2]}',
                f'Dummy__{perm[3]}')
    assert not covc2.spin0
    cov2 = covc2.get_covariance()
    assert cov2.shape == (ncls1*nbpw, ncls2*nbpw)
    shutil.rmtree(tmpdir1)

    # Spin-0 covariance class
    data['cov']['spin0'] = True
    covc0 = Cov(data,
                f'Dummy__{perm[0]}',
                f'Dummy__{perm[1]}',
                f'Dummy__{perm[2]}',
                f'Dummy__{perm[3]}')
    assert covc0.spin0
    cov0 = covc0.get_covariance()
    assert cov0.shape == (ncls1*nbpw, ncls2*nbpw)
    shutil.rmtree(tmpdir1)

    if ncls1 == ncls2:
        # Check that they are the same on all bandpowers
        # except the first one
        r = np.diag(cov2)/np.diag(cov0)-1
        # This loops through EE, EB, BE, BB
        for i in range(ncls1):
            assert np.all(r[i::ncls1][1:] < 1E-5)


@pytest.mark.parametrize('tr1,tr2', [('galaxy_density', 'galaxy_density'),
                                     ('galaxy_density', 'galaxy_shear'),
                                     ('galaxy_density', 'cmb_convergence'),
                                     ('galaxy_shear', 'galaxy_shear'),
                                     ('galaxy_shear', 'cmb_convergence'),
                                     ('cmb_convergence', 'cmb_convergence')])
def test_clfid_against_ccl(tr1, tr2):
    data = get_config(dtype0=tr1, dtype1=tr2)
    if tr1 == 'galaxy_density':
        data['tracers']['Dummy__0']['bias'] = 1.
    elif tr1 == 'galaxy_shear':
        data['tracers']['Dummy__0']['m'] = 0.
    if tr2 == 'galaxy_density':
        data['tracers']['Dummy__1']['bias'] = 1.
    elif tr2 == 'galaxy_shear':
        data['tracers']['Dummy__1']['m'] = 0.

    cosmo = ccl.Cosmology(**data['cov']['fiducial']['cosmo'])
    clf = ClFid(data, 'Dummy__0', 'Dummy__1')
    d = clf.get_cl_file()
    shutil.rmtree(tmpdir1)

    def get_ccl_tracer(tr):
        if tr == 'galaxy_density':
            z, nz = np.loadtxt('xcell/tests/data/DESY1gc_dndz_bin0.txt',
                               usecols=(1, 3), unpack=True)
            t = ccl.NumberCountsTracer(cosmo, False, dndz=(z, nz),
                                       bias=(z, np.ones_like(z)))
        elif tr == 'galaxy_shear':
            z, nz = np.loadtxt('xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
                               usecols=(0, 1), unpack=True)
            t = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
        elif tr == 'cmb_convergence':
            t = ccl.CMBLensingTracer(cosmo, z_source=1100.)
        return t

    t1 = get_ccl_tracer(tr1)
    t2 = get_ccl_tracer(tr2)
    clb = ccl.angular_cl(cosmo, t1, t2, d['ell'])

    assert np.all(np.fabs(clb[2:]/d['cl'][0][2:]-1) < 1E-5)
