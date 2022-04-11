import shutil
import os
import numpy as np
from xcell.cls.theory import ConcentrationDuffy08M500c
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
               dtype1='galaxy_density',
               inc_hm=False):
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
              'cosmo': cosmo, 'fsky': fsky, 'seed': 0,
              'dtype': dtype0, 'use_halo_model': inc_hm}
    dummy1 = {'mask_name': 'mask_dummy1', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'fsky': fsky2, 'seed': 100,
              'dtype': dtype1, 'use_halo_model': inc_hm}
    dummy2 = {'mask_name': 'mask_dummy2', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'fsky': fsky2, 'seed': 100,
              'dtype': dtype1, 'use_halo_model': inc_hm, 'mask_power': 2}
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0, 'Dummy__1': dummy1,
                        'Dummy__2': dummy2},
            'cls': {'Dummy-Dummy': {'compute': 'all'}},
            'cov': {'fiducial': {'cosmo': cosmo,
                                 'wl_m': False, 'wl_ia': False}},
            'bpw_edges': bpw_edges,
            'sphere': {'n_iter_sht': 0,
                       'n_iter_mcm': 3,
                       'n_iter_cmcm': 3,
                       'nside': nside,
                       'coords': 'C'},
            'recompute': {'cls': True,
                          'cov': True,
                          'mcm': True,
                          'cmcm': True},
            'output': tmpdir1}


def get_cl_class(fsky=0.2, fiducial=False):
    data = get_config(fsky)
    if fiducial:
        return ClFid(data, 'Dummy__0', 'Dummy__0')
    else:
        return Cl(data, 'Dummy__0', 'Dummy__0')


def get_cov_class(fsky=0.2):
    data = get_config(fsky)
    return Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')


def test_smoke():
    get_cl_class()
    get_cov_class()
    shutil.rmtree(tmpdir1)


def test_cl_correction():
    data = get_config()
    cl_class = Cl(data, 'Dummy__2', 'Dummy__2')
    cl_file = cl_class.get_cl_file()
    correct = cl_file['correction']
    dummy = MapperDummy(data['tracers']['Dummy__2'])
    w_a = dummy.get_mask()
    w_b = dummy.get_mask()
    n_a = dummy.mask_power
    n_b = dummy.mask_power
    correct_b = np.mean(w_a*w_b)/np.mean(w_a**n_a*w_b**n_b)
    assert correct != 1
    assert correct == correct_b


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


def test_cov_ng_error():
    data = get_config(fsky=0.2)
    covc = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    with pytest.raises(NotImplementedError):
        covc.get_covariance_ng_halomodel(0, 0, 0, 0, 0.2, kind='3h')
    shutil.rmtree(tmpdir1)


def test_cov_ng_1h():
    # From CCL directly
    data = get_config(fsky=0.2)
    clc = Cl(data, 'Dummy__0', 'Dummy__0')
    ells = clc.b.get_effective_ells()
    shutil.rmtree(tmpdir1)
    cosmo = ccl.Cosmology(**data['cov']['fiducial']['cosmo'])
    md = ccl.halos.MassDef200m()
    mf = ccl.halos.MassFuncTinker10(cosmo, mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mdef=md)
    hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)
    pr = ccl.halos.HaloProfileHOD(cm, lMmin_0=12.1,
                                  lM1_p=0.1, bg_0=1.2)
    prof2pt = ccl.halos.Profile2ptHOD()
    z, nz = np.loadtxt('xcell/tests/data/DESY1gc_dndz_bin0.txt',
                       usecols=(1, 3), unpack=True)
    tr = ccl.NumberCountsTracer(cosmo, False, dndz=(z, nz),
                                bias=(z, np.ones_like(z)))
    k_arr = np.geomspace(1E-4, 1E2, 256)
    a_arr = 1./(1+np.linspace(0, 3, 15)[::-1])
    tkk = ccl.halos.halomod_Tk3D_1h(cosmo, hmc,
                                    prof1=pr, prof2=pr, prof12_2pt=prof2pt,
                                    prof3=pr, prof4=pr, prof34_2pt=prof2pt,
                                    normprof1=True, normprof2=True,
                                    normprof3=True, normprof4=True,
                                    a_arr=a_arr, lk_arr=np.log(k_arr))
    covNG0 = ccl.angular_cl_cov_cNG(cosmo,
                                    cltracer1=tr, cltracer2=tr,
                                    ell=ells, tkka=tkk, fsky=1.,
                                    cltracer3=tr, cltracer4=tr, ell2=ells)

    # Gaussian only
    data = get_config(fsky=0.2, inc_hm=True)
    data['tracers']['Dummy__0']['hod_params'] = {'lMmin_0': 12.1,
                                                 'lM1_p': 0.1,
                                                 'bg_0': 1.2}
    covcG = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    covG = covcG.get_covariance()
    shutil.rmtree(tmpdir1)

    # Gaussian + non-Gaussian
    data = get_config(fsky=0.2, inc_hm=True)
    data['cov']['non_Gaussian'] = True
    data['cov']['NG_terms'] = ['1h']
    data['tracers']['Dummy__0']['hod_params'] = {'lMmin_0': 12.1,
                                                 'lM1_p': 0.1,
                                                 'bg_0': 1.2}
    covc1 = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    mapper = MapperDummy(data['tracers']['Dummy__0'])
    fsky = np.mean((mapper.get_mask() > 0))
    covNG1 = covc1.get_covariance_ng_halomodel(0, 0, 0, 0, fsky)
    cov1 = covc1.get_covariance()
    shutil.rmtree(tmpdir1)

    # fsky on input
    data = get_config(fsky=0.2, inc_hm=True)
    data['cov']['non_Gaussian'] = True
    data['cov']['NG_terms'] = ['1h']
    data['cov']['fsky_NG'] = 0.1
    data['tracers']['Dummy__0']['hod_params'] = {'lMmin_0': 12.1,
                                                 'lM1_p': 0.1,
                                                 'bg_0': 1.2}
    covc2 = Cov(data, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    covNG2 = covc2.get_covariance()-covG
    shutil.rmtree(tmpdir1)

    # Tests
    # Compare result of NG method with G+NG-G
    assert np.allclose(covNG1, cov1-covG, atol=0)
    # Compare with CCL prediction
    # (interpolation errors are ~1E-4)
    assert np.allclose(covNG0, covNG1*fsky, atol=0, rtol=1E-3)
    # fsky scaling
    assert np.allclose(covNG2, covNG1*fsky/0.1, atol=0)


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


def test_custom_auto():
    # No custom auto
    data = get_config()
    clc1 = Cl(data, 'Dummy__0', 'Dummy__0')
    l1, cl1 = clc1.get_ell_cl_cp()
    shutil.rmtree(tmpdir1)

    # With custom auto
    data = get_config()
    data['tracers']['Dummy__0']['custom_auto'] = True
    data['tracers']['Dummy__0']['custom_offset'] = np.pi*1E-5
    clc2 = Cl(data, 'Dummy__0', 'Dummy__0')
    l2, cl2 = clc2.get_ell_cl_cp()
    shutil.rmtree(tmpdir1)

    assert np.allclose(cl1, cl2-np.pi*1E-5, rtol=1E-4, atol=0)

    # Covariance custom cross
    data = get_config()
    data['tracers']['Dummy__0']['custom_auto'] = True
    data['tracers']['Dummy__0']['custom_offset'] = np.pi*1E-5
    clc3 = Cl(data, 'Dummy__0', 'Dummy__0')
    l2, cl3 = clc3.get_ell_cl_cp_cov()
    shutil.rmtree(tmpdir1)

    assert np.allclose(cl1, cl3, rtol=1E-4, atol=0)


def test_get_ell_cl_cp():
    # Get cl from map
    cl_class = get_cl_class()
    ell, cl = cl_class.get_ell_cl()
    ell, cl_cp = cl_class.get_ell_cl_cp()

    w = cl_class.get_workspace()
    cl2 = w.decouple_cell(cl_cp)
    shutil.rmtree(tmpdir1)
    assert np.all(np.fabs(cl / cl2 - 1) < 1e-10)

    # Test it also in ClFid
    cl_class = get_cl_class(fiducial=True)
    ell, cl = cl_class.get_ell_cl()
    ell, cl_cp = cl_class.get_ell_cl_cp()
    cl_cp2 = w.couple_cell(cl)
    shutil.rmtree(tmpdir1)
    assert np.all(np.fabs(cl_cp / cl_cp2 - 1) < 1e-10)


def test_get_ell_cl_binned():
    # Get cl from map
    cl_class = get_cl_class()
    ell = cl_class.get_ell_cl()[0]
    w = cl_class.get_workspace()

    # Test it also in ClFid
    cl_class = get_cl_class(fiducial=True)
    cl = cl_class.get_ell_cl()[1]
    ell_binned, cl_binned = cl_class.get_ell_cl_binned()
    cl_binned2 = w.decouple_cell(w.couple_cell(cl))
    shutil.rmtree(tmpdir1)
    assert np.all(np.fabs(cl_binned / cl_binned2 - 1) < 1e-10)
    assert np.all(np.fabs(ell / ell_binned - 1) < 1e-10)


def test_covar_from_data():
    config = get_config(dtype0='generic')
    # Can't compute covariance unless we allow doing it from data
    with pytest.raises(NotImplementedError):
        Cov(config, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    shutil.rmtree(tmpdir1)

    # Allow falling back to data
    config = get_config(dtype0='generic')
    config['cov']['data_fallback'] = True
    cov_obj = Cov(config, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    cov1 = cov_obj.get_covariance()
    shutil.rmtree(tmpdir1)

    # Compute from data on purpose
    config = get_config(dtype0='generic')
    config['cov']['cls_from_data'] = "all"
    cov_obj = Cov(config, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    cov2 = cov_obj.get_covariance()
    shutil.rmtree(tmpdir1)

    assert np.allclose(cov1, cov2, atol=1E-10, rtol=0)


@pytest.mark.parametrize('cldata', ['all', 'none'])
def test_get_covariance(cldata):
    # Get cl from randomnly generated map ("data")
    config = get_config(fsky=1)
    config['cov']['cls_from_data'] = cldata

    cl_class = Cl(config, 'Dummy__0', 'Dummy__0')
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
    cov_class = Cov(config, 'Dummy__0', 'Dummy__0', 'Dummy__0', 'Dummy__0')
    cov = cov_class.get_covariance()

    diag = (2 * cl_m1[0]**2) / (2 * ell + 1) / 4
    cov_m = np.diag(diag)

    icov = np.linalg.inv(cov)
    icov_m = np.linalg.inv(cov_m)
    dCl = (cl_data - cl_m1)[0]

    chi2 = dCl.dot(icov).dot(dCl)
    chi2_m = dCl.dot(icov_m).dot(dCl)

    shutil.rmtree(tmpdir1)
    assert np.fabs(chi2/chi2_m-1) < 0.03


def test_cls_vs_namaster():
    # cls
    # Get cl from randomnly generated map ("data")
    cl_class = get_cl_class()
    ell, cl_data = cl_class.get_ell_cl()
    b = cl_class.get_NmtBin()
    win = cl_class.get_bandpower_windows()
    # Read output
    clfile = np.load(os.path.join(tmpdir1,
                                  'Dummy_Dummy',
                                  'cl_Dummy__0_Dummy__0.npz'))
    shutil.rmtree(tmpdir1)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()
    shutil.rmtree(tmpdir1)

    # NaMaster
    config = get_config()
    conf = config['tracers']['Dummy__0']
    conf['nside'] = config['sphere']['nside']
    conf['coords'] = config['sphere']['coords']
    m = MapperDummy(conf)

    # True cl
    cl_m = m.get_cl()
    spin = m.get_spin()
    mask = m.get_mask()
    signal_map = m.get_signal_map()
    n_iter_sht = config['sphere']['n_iter_sht']
    n_iter_mcm = config['sphere']['n_iter_mcm']
    n_iter_cmcm = config['sphere']['n_iter_cmcm']
    # Compute Cl from map
    f = nmt.NmtField(mask, signal_map, spin=spin, n_iter=n_iter_sht)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(f, f, bins=b, n_iter=n_iter_mcm)
    cl_data_nmt_cp = nmt.compute_coupled_cell(f, f)
    cl_data_nmt = wsp.decouple_cell(cl_data_nmt_cp)

    # Couple true Cl
    cl_m_cp = wsp.couple_cell([cl_m])
    cl_m = wsp.decouple_cell(cl_m_cp)

    # Compute cov with NaMaster
    cwsp = nmt.NmtCovarianceWorkspace()
    cwsp.compute_coupling_coefficients(f, f, n_iter=n_iter_cmcm)
    cl_cov = cl_m_cp / np.mean(mask * mask)
    cov_nmt = nmt.gaussian_covariance(cwsp, spin, spin, spin, spin, cl_cov,
                                      cl_cov, cl_cov, cl_cov, wsp)
    bpwin = wsp.get_bandpower_windows()
    icov_nmt = np.linalg.inv(cov_nmt)

    def compare(cl, cv, wn, tol=1E-5):
        rdev = cl / cl_data_nmt - 1
        assert np.max(np.abs(rdev)) < tol

        # Compare cl and covariance
        icov = np.linalg.inv(cv)
        dCl = (cl - cl_m)[0]
        chi2 = dCl.dot(icov).dot(dCl)
        chi2_m = dCl.dot(icov_nmt).dot(dCl)
        assert np.fabs(chi2/chi2_m-1) < tol

        # Compare bandpower windows
        assert np.all(win == bpwin)

    compare(cl_data, cov, win)
    compare(clfile['cl'], cov, clfile['wins'])
    assert np.allclose(clfile['cl_cp'], cl_data_nmt_cp, atol=0)
    assert np.allclose(clfile['cl_cov_cp'], cl_data_nmt_cp, atol=0)
    assert np.allclose(clfile['cl_cov_11_cp'], cl_data_nmt_cp, atol=0)
    assert np.allclose(clfile['cl_cov_12_cp'], cl_data_nmt_cp, atol=0)
    assert np.allclose(clfile['cl_cov_22_cp'], cl_data_nmt_cp, atol=0)


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


def test_ignore_existing_yml():
    # Test for Cls
    cl_class = get_cl_class()
    data = cl_class.data.data
    # Now, data['cls']['Dummy-Dummy']['compute'] = 'all'. We change it to
    # 'auto' and check that is read when ignore_existing_yml=True
    data['cls']['Dummy-Dummy']['compute'] = 'auto'
    cl_class01 = Cl(data, 'Dummy__0', 'Dummy__1', ignore_existing_yml=True)
    assert cl_class01.data.data['cls']['Dummy-Dummy']['compute'] == 'auto'
    cl_class01 = Cl(data, 'Dummy__0', 'Dummy__1', ignore_existing_yml=False)
    assert cl_class01.data.data['cls']['Dummy-Dummy']['compute'] == 'all'

    # Test for Cov
    cov_class = Cov(data, 'Dummy__0', 'Dummy__1', 'Dummy__0', 'Dummy__1',
                    ignore_existing_yml=True)
    assert cov_class.data.data['cls']['Dummy-Dummy']['compute'] == 'auto'
    cov_class = Cov(data, 'Dummy__0', 'Dummy__1', 'Dummy__0', 'Dummy__1',
                    ignore_existing_yml=False)
    assert cov_class.data.data['cls']['Dummy-Dummy']['compute'] == 'all'
    shutil.rmtree(tmpdir1)


def test_unsupported_quantity():
    data = get_config(dtype0='generic')
    with pytest.raises(NotImplementedError):
        ClFid(data, 'Dummy__0', 'Dummy__1')
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


def test_clfid_halomod_settings():
    data = get_config()

    # Empty halo model parameters (default values)
    clf = ClFid(data, 'Dummy__0', 'Dummy__1')
    cosmo = clf.th.get_cosmo_ccl()
    hm_par = clf.th.get_halomodel_params()
    assert np.fabs(hm_par['mass_def'].get_Delta(cosmo, 1.)
                   - 200) < 1E-3
    assert hm_par['mass_def'].rho_type == 'matter'
    assert hm_par['mass_func'].name == 'Tinker10'
    assert hm_par['halo_bias'].name == 'Tinker10'
    assert hm_par['cM'].name == 'Duffy08'
    shutil.rmtree(tmpdir1)

    # Custom halo model parameters
    md = '200c'
    mf = 'Tinker08'
    hb = 'Tinker10'
    cM = 'Bhattacharya13'
    data['cov']['fiducial']['halo_model'] = {'mass_def': md,
                                             'mass_function': mf,
                                             'halo_bias': hb,
                                             'concentration': cM}
    clf = ClFid(data, 'Dummy__0', 'Dummy__1')
    cosmo = clf.th.get_cosmo_ccl()
    hm_par = clf.th.get_halomodel_params()
    assert np.fabs(hm_par['mass_def'].get_Delta(cosmo, 1.)
                   - 200) < 1E-3
    assert hm_par['mass_def'].rho_type == 'critical'
    assert hm_par['mass_func'].name == mf
    assert hm_par['halo_bias'].name == hb
    assert hm_par['cM'].name == cM
    shutil.rmtree(tmpdir1)


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
        data['tracers']['Dummy__0']['magnif_s'] = 1
    elif tr1 == 'galaxy_shear':
        data['tracers']['Dummy__0']['m'] = 0.
    if tr2 == 'galaxy_density':
        data['tracers']['Dummy__1']['bias'] = 1.
        data['tracers']['Dummy__1']['magnif_s'] = 1
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
                                       bias=(z, np.ones_like(z)),
                                       mag_bias=(z, np.ones_like(z)))
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


@pytest.mark.parametrize('tr1,tr2', [('galaxy_shear', 'galaxy_shear'),
                                     ('galaxy_density', 'galaxy_shear'),
                                     ('cmb_tSZ', 'galaxy_shear'),
                                     ('cmb_tSZ', 'cmb_convergence')])
def test_clfid_halomod(tr1, tr2):
    data = get_config(dtype0=tr1, dtype1=tr2, inc_hm=True)

    cosmo = ccl.Cosmology(**data['cov']['fiducial']['cosmo'])
    md = ccl.halos.MassDef200m()
    mf = ccl.halos.MassFuncTinker10(cosmo, mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mdef=md)
    hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)
    pNFW = ccl.halos.HaloProfileNFW(cm)
    profs = {}
    ccltr = {}
    normed = {}
    for tr, lab in [(tr1, 'Dummy__0'), (tr2, 'Dummy__1')]:
        if tr == 'galaxy_density':
            data['tracers'][lab]['hod_params'] = {'lMmin_0': 12.1,
                                                  'lM1_p': 0.1,
                                                  'bg_0': 1.2}
            profs[tr] = ccl.halos.HaloProfileHOD(cm, lMmin_0=12.1,
                                                 lM1_p=0.1, bg_0=1.2)
            z, nz = np.loadtxt('xcell/tests/data/DESY1gc_dndz_bin0.txt',
                               usecols=(1, 3), unpack=True)
            ccltr[tr] = ccl.NumberCountsTracer(cosmo, False, dndz=(z, nz),
                                               bias=(z, np.ones_like(z)))
            normed[tr] = True
        elif tr == 'cmb_tSZ':
            data['tracers'][lab]['gnfw_params'] = {'mass_bias': 0.9}
            profs[tr] = ccl.halos.HaloProfilePressureGNFW(mass_bias=0.9)
            ccltr[tr] = ccl.tSZTracer(cosmo, z_max=3.)
            normed[tr] = False
        elif tr == 'galaxy_shear':
            profs[tr] = pNFW
            z, nz = np.loadtxt('xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
                               usecols=(0, 1), unpack=True)
            ccltr[tr] = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
            normed[tr] = True
        elif tr == 'cmb_convergence':
            profs[tr] = pNFW
            ccltr[tr] = ccl.CMBLensingTracer(cosmo, z_source=1100.)
            normed[tr] = True

    clf = ClFid(data, 'Dummy__0', 'Dummy__1')
    d = clf.get_cl_file()
    shutil.rmtree(tmpdir1)

    k_arr = np.geomspace(1E-4, 1E2, 512)
    a_arr = 1./(1+np.linspace(0, 3, 15)[::-1])
    pk = ccl.halos.halomod_Pk2D(cosmo, hmc, profs[tr1],
                                prof2=profs[tr2],
                                normprof1=normed[tr1],
                                normprof2=normed[tr2],
                                lk_arr=np.log(k_arr),
                                a_arr=a_arr)
    # Commented out until these features are pushed to the pip release of CCL
    # smooth_transition=(lambda a: 0.7),
    # supress_1h=(lambda a: 0.01))
    clb = ccl.angular_cl(cosmo, ccltr[tr1], ccltr[tr2], d['ell'], p_of_k_a=pk)

    assert np.all(np.fabs(clb[2:]/d['cl'][0][2:]-1) < 1E-4)


def test_clfid_halomod_M500c():
    tr1 = 'cmb_tSZ'
    tr2 = 'cmb_convergence'
    data = get_config(dtype0=tr1, dtype1=tr2, inc_hm=True)
    data['cov']['fiducial']['halo_model'] = {'mass_def': '500c',
                                             'concentration': 'Duffy08M500c'}
    data['tracers']['Dummy__0']['gnfw_params'] = {'mass_bias': 0.9}

    cosmo = ccl.Cosmology(**data['cov']['fiducial']['cosmo'])
    md = ccl.halos.MassDef(500, 'critical')
    mf = ccl.halos.MassFuncTinker10(cosmo, mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=md)
    cm = ConcentrationDuffy08M500c(mdef=md)
    hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)
    prof1 = ccl.halos.HaloProfilePressureGNFW(mass_bias=0.9)
    ccltr1 = ccl.tSZTracer(cosmo, z_max=3.)
    prof2 = ccl.halos.HaloProfileNFW(cm)
    ccltr2 = ccl.CMBLensingTracer(cosmo, z_source=1100.)

    clf = ClFid(data, 'Dummy__0', 'Dummy__1')
    d = clf.get_cl_file()
    shutil.rmtree(tmpdir1)

    k_arr = np.geomspace(1E-4, 1E2, 512)
    a_arr = 1./(1+np.linspace(0, 6., 30)[::-1])
    pk = ccl.halos.halomod_Pk2D(cosmo, hmc, prof1,
                                prof2=prof2,
                                normprof1=False,
                                normprof2=True,
                                lk_arr=np.log(k_arr),
                                a_arr=a_arr)
    # Commented out until these features are pushed to the pip release of CCL
    # smooth_transition=(lambda a: 0.7),
    # supress_1h=(lambda a: 0.01))
    clb = ccl.angular_cl(cosmo, ccltr1, ccltr2, d['ell'], p_of_k_a=pk)

    assert np.all(np.fabs(clb[2:]/d['cl'][0][2:]-1) < 1E-4)
