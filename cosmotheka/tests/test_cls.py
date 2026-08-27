import shutil
import os
import numpy as np
from cosmotheka.cls.theory import ConcentrationDuffy08M500c
from cosmotheka.cls.cl import Cl, ClFid
from cosmotheka.cls.cov import Cov
import pymaster as nmt
from cosmotheka.mappers import MapperDummy
import pytest
import pyccl as ccl
import healpy as hp


# Remove previous test results
tmpdir1 = "./cosmotheka/tests/cls/dummy1"
tmpdir2 = "./cosmotheka/tests/cls/dummy2"
NSIDE = 32


def setup_module():
    os.makedirs(tmpdir1, exist_ok=True)
    os.makedirs(tmpdir2, exist_ok=True)


# Cleaning the tmp dir before running and after running the tests
def _clean_tmpdir(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def _clean_tmpdirs():
    for path in (tmpdir1, tmpdir2):
        _clean_tmpdir(path)


def teardown_module():
    _clean_tmpdirs()


@pytest.fixture(autouse=True)
def run_clean_tmp():
    _clean_tmpdirs()   # before each test
    yield
    _clean_tmpdirs()   # after each test


def get_config(
    fsky=0.2,
    fsky2=0.3,
    dtype0="galaxy_density",
    dtype1="galaxy_density",
    inc_hm=False,
):
    # Set only the necessary entries. Leave the others to their default value.
    cosmo = {
        # Planck 2018: Table 2 of 1807.06209
        # Omega_m: 0.3133
        "Omega_c": 0.2640,
        "Omega_b": 0.0493,
        "h": 0.6736,
        "n_s": 0.9649,
        "sigma8": 0.8111,
        "w0": -1,
        "wa": 0,
        "transfer_function": "eisenstein_hu",
        "baryonic_effects": None,
    }
    dummy0 = {
        "mask_name": "mask_dummy0",
        "mapper_class": "MapperDummy",
        "cosmo": cosmo,
        "fsky": fsky,
        "seed": 0,
        "dtype": dtype0,
        "use_halo_model": inc_hm,
    }
    dummy1 = {
        "mask_name": "mask_dummy1",
        "mapper_class": "MapperDummy",
        "cosmo": cosmo,
        "fsky": fsky2,
        "seed": 100,
        "dtype": dtype1,
        "use_halo_model": inc_hm,
    }
    dummy2 = {
        "mask_name": "mask_dummy2",
        "mapper_class": "MapperDummy",
        "cosmo": cosmo,
        "fsky": fsky2,
        "seed": 100,
        "dtype": dtype1,
        "use_halo_model": inc_hm,
        "mask_power": 2,
    }
    bpw_edges = list(range(0, 3 * NSIDE, 4))

    out = {
        "tracers": {
            "Dummy__0": dummy0,
            "Dummy__1": dummy1,
            "Dummy__2": dummy2,
        },
        "cls": {"Dummy-Dummy": {"compute": "all"}},
        "cov": {"fiducial": {"cosmo": cosmo, "wl_ia": False}},
        "bpw_edges": bpw_edges,
        "sphere": {
            "n_iter_sht": 0,
            "n_iter_mcm": 3,
            "n_iter_cmcm": 3,
            "nside": NSIDE,
            "coords": "C",
        },
        "recompute": {"cls": True, "cov": True, "mcm": True, "cmcm": True},
        "output": tmpdir1,
    }

    if dtype0 == "cmb_convergence" or dtype1 == "cmb_convergence":
        out["cls"]["Dummy-Dummy"]["neglect_mc_correction"] = True

    return out


def get_cl_class(fsky=0.2, fiducial=False):
    data = get_config(fsky)
    if fiducial:
        return ClFid(data, "Dummy__0", "Dummy__0")
    else:
        return Cl(data, "Dummy__0", "Dummy__0")


def get_cov_class(fsky=0.2):
    data = get_config(fsky)
    return Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")


def test_smoke():
    get_cl_class()
    get_cov_class()
    # _clean_tmpdir(tmpdir1)


def test_cl_and_cov_file_expected_entries():
    data = get_config()

    cl_class = Cl(data, "Dummy__0", "Dummy__0")
    cl_class.get_cl_file()
    cov_class = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    cov_class.get_covariance()

    cl_path = os.path.join(
        tmpdir1, "Dummy_Dummy", "cl_Dummy__0_Dummy__0.npz"
    )
    cov_path = os.path.join(
        tmpdir1, "cov", "cov_Dummy__0_Dummy__0_Dummy__0_Dummy__0.npz"
    )

    cl_file = np.load(cl_path)
    cov_file = np.load(cov_path)

    expected_cl_keys = {
        "ell",
        "cl",
        "cl_cp",
        "nl",
        "nl_cp",
        "cl_cov",
        "cl_cov_11",
        "cl_cov_12",
        "cl_cov_22",
        "wins",
        "correction",
        "mean_mamb",
        "crude_err",
        "threshold",
    }
    assert set(cl_file.files) == expected_cl_keys

    expected_cov_keys = {
        "cov",
        "cov_G",
        "cov_NG",
        "cov_SSC",
        "cov_nl_marg",
        "cov_m_marg",
        "cov_NG_1h",
        "cov_NG_2h",
        "cov_NG_3h",
        "cov_NG_4h",
        "threshold",
        "notnull",
    }
    assert set(cov_file.files) == expected_cov_keys


def test_cl_correction():
    data = get_config()
    cl_class = Cl(data, "Dummy__2", "Dummy__2")
    cl_file = cl_class.get_cl_file()
    correct = cl_file["correction"]
    dummy = MapperDummy(data["tracers"]["Dummy__2"])
    w_a = dummy.get_mask()
    w_b = dummy.get_mask()
    n_a = dummy.mask_power
    n_b = dummy.mask_power
    correct_b = np.mean(w_a * w_b) / np.mean(w_a**n_a * w_b**n_b)
    assert correct != 1
    assert correct == correct_b


def test_get_nmtbin():
    # This test checks that the bpw_edges can be read from the
    # global part of the yaml file or from the context of one
    # of the cross-correlations.
    # 1. From global
    data = get_config()
    cl1 = Cl(data, "Dummy__0", "Dummy__0")
    _clean_tmpdir(tmpdir1)
    # 2. From cross-correlations
    data["cls"]["Dummy-Dummy"]["bpw_edges"] = data.pop("bpw_edges")
    cl2 = Cl(data, "Dummy__0", "Dummy__0")
    _clean_tmpdir(tmpdir1)
    # Check they are the same
    b1 = cl1.get_NmtBin()
    b2 = cl2.get_NmtBin()
    assert np.all(b1.get_effective_ells() == b2.get_effective_ells())


def test_cov_nlmarg():
    data = get_config(0.2)
    data["tracers"]["Dummy__0"]["nl_marginalize"] = True
    data["tracers"]["Dummy__0"]["nl_prior"] = 1e30
    data["tracers"]["Dummy__0"]["noise_level"] = 1e-5
    data["cov"]["error_threshold"] = 1e100
    data["output"] = tmpdir2
    cov_class = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    cov = cov_class.get_covariance()
    num_l = len(cov)
    oo = np.ones(num_l)
    chi2 = np.dot(oo, np.linalg.solve(cov, oo))
    assert np.fabs(chi2) < 1e-5 * num_l
    _clean_tmpdir(tmpdir2)

    # The prior is huge so check that it will fail if the error_threshold is
    # not set (i.e. it is based on an estimation from data)
    with pytest.raises(RuntimeError):
        del data["cov"]["error_threshold"]
        cov_class = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
        cov = cov_class.get_covariance()
    _clean_tmpdir(tmpdir2)


# TODO: Missing test for:
#       a) 1h+2h combinations
#       b) computation of missing terms. E.g. 1h is already in but the user
#          requests 1h+2h later on. It only computes 2h and add it to the 1h
#          contribution already there.
@pytest.mark.skip("slow")
@pytest.mark.parametrize("kind", ["1h", "2h", "3h", "4h", None])
def test_cov_ng(kind):
    # From CCL directly
    data = get_config(fsky=0.2)
    clc = Cl(data, "Dummy__0", "Dummy__0")
    ells = clc.b.get_effective_ells()
    _clean_tmpdir(tmpdir1)
    cosmo = ccl.Cosmology(**data["cov"]["fiducial"]["cosmo"])
    md = ccl.halos.MassDef200m
    mf = ccl.halos.MassFuncTinker10(mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mass_def=md)
    hmc = ccl.halos.HMCalculator(mass_function=mf, halo_bias=hb, mass_def=md)
    pr = ccl.halos.HaloProfileHOD(
        mass_def=md,
        concentration=cm,
        log10Mmin_0=12.1,
        log10M1_p=0.1,
        bg_0=1.2,
    )
    prof2pt = ccl.halos.Profile2ptHOD()
    z, nz = np.loadtxt(
        "cosmotheka/tests/data/DESY1gc_dndz_bin0.txt",
        usecols=(1, 3),
        unpack=True,
    )
    tr = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz), bias=(z, np.ones_like(z))
    )
    # In order to get rdev = 1e-3, we need to have an a_arr, and k_arr very
    # close to the one in ClFid.
    # In the code, to speed things up, we remove half of the points

    lk = cosmo.get_pk_spline_lk()
    sel = lk <= np.log(100)
    k_arr = np.exp(lk[sel])[::2]

    a_arr = np.linspace(1/(1+6), 1, 38)[::2]
    if a_arr[-1] != 1.0:
        a_arr = np.append(a_arr, 1.0)
    separable_growth = True

    if kind == "1h":
        tkk = ccl.halos.halomod_Tk3D_1h(
            cosmo,
            hmc,
            prof=pr,
            prof12_2pt=prof2pt,
            a_arr=a_arr,
            lk_arr=np.log(k_arr),
        )
    elif kind == "2h":
        tkk = ccl.halos.halomod_Tk3D_2h(
            cosmo,
            hmc,
            prof=pr,
            prof12_2pt=prof2pt,
            a_arr=a_arr,
            lk_arr=np.log(k_arr),
            separable_growth=separable_growth,
        )
    elif kind == "3h":
        tkk = ccl.halos.halomod_Tk3D_3h(
            cosmo,
            hmc,
            prof=pr,
            prof13_2pt=prof2pt,
            a_arr=a_arr,
            lk_arr=np.log(k_arr),
            separable_growth=separable_growth,
        )
    elif kind == "4h":
        tkk = ccl.halos.halomod_Tk3D_4h(
            cosmo,
            hmc,
            prof=pr,
            a_arr=a_arr,
            lk_arr=np.log(k_arr),
            separable_growth=separable_growth,
        )
    elif kind is None:
        tkk = ccl.halos.halomod_Tk3D_cNG(
            cosmo=cosmo,
            hmc=hmc,
            prof=pr,
            prof12_2pt=prof2pt,
            a_arr=a_arr,
            lk_arr=np.log(k_arr),
            separable_growth=separable_growth,
        )

    covNG0 = ccl.angular_cl_cov_cNG(
        cosmo,
        tracer1=tr,
        tracer2=tr,
        ell=ells,
        t_of_kk_a=tkk,
        fsky=1.0,
        tracer3=tr,
        tracer4=tr,
        ell2=ells,
    )

    # Gaussian only
    data = get_config(fsky=0.2, inc_hm=True)
    data["tracers"]["Dummy__0"]["hod_params"] = {
        "log10Mmin_0": 12.1,
        "log10M1_p": 0.1,
        "bg_0": 1.2,
    }
    covcG = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    covG = covcG.get_covariance()
    _clean_tmpdir(tmpdir1)

    # Gaussian + non-Gaussian
    data = get_config(fsky=0.2, inc_hm=True)
    data["cov"]["non_Gaussian"] = {'compute': True, 'NG_terms': [kind]}
    data["tracers"]["Dummy__0"]["hod_params"] = {
        "log10Mmin_0": 12.1,
        "log10M1_p": 0.1,
        "bg_0": 1.2,
    }
    covc1 = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    mapper = MapperDummy(data["tracers"]["Dummy__0"])
    fsky = np.mean((mapper.get_mask() > 0))
    covNG1 = covc1.get_covariance_ng_halomodel(0, 0, 0, 0, kind=kind)
    cov1 = covc1.get_covariance()
    _clean_tmpdir(tmpdir1)

    # fsky on input
    data = get_config(fsky=0.2, inc_hm=True)
    data["cov"]["non_Gaussian"] = {'compute': True, 'NG_terms': [kind]}
    data["cov"]["fsky_NG"] = 0.1
    data["tracers"]["Dummy__0"]["hod_params"] = {
        "log10Mmin_0": 12.1,
        "log10M1_p": 0.1,
        "bg_0": 1.2,
    }
    covc2 = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    covNG2 = covc2.get_covariance() - covG
    _clean_tmpdir(tmpdir1)

    # Tests (using pytest.approx for more informative logs)
    # Compare result of NG method with G+NG-G
    assert covNG1 == pytest.approx(cov1 - covG, rel=1e-4, abs=0)
    # Compare with CCL prediction
    # (interpolation errors are ~1E-4, using a very similar grid)
    assert covNG0 == pytest.approx(covNG1 * fsky, rel=1e-3, abs=0)
    # fsky scaling
    assert covNG2 == pytest.approx(covNG1 * fsky / 0.1, rel=1e-4, abs=0)


@pytest.mark.skip("slow")
@pytest.mark.parametrize("sigma2B_type", ["fsky", "mask_wl"])
def test_cov_ssc(sigma2B_type):
    # From CCL directly
    data = get_config(fsky=0.2)
    clc = Cl(data, "Dummy__0", "Dummy__0")
    ells = clc.b.get_effective_ells()
    mapper, _ = clc.get_mappers()
    mask = mapper.get_mask()
    _clean_tmpdir(tmpdir1)

    cosmo = ccl.Cosmology(**data["cov"]["fiducial"]["cosmo"])
    md = ccl.halos.MassDef200m
    mf = ccl.halos.MassFuncTinker10(mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mass_def=md)
    hmc = ccl.halos.HMCalculator(mass_function=mf, halo_bias=hb, mass_def=md)
    prnfw = ccl.halos.HaloProfileNFW(mass_def=md, concentration=cm)
    z, nz = np.loadtxt(
        "cosmotheka/tests/data/DESY1gc_dndz_bin0.txt",
        usecols=(1, 3),
        unpack=True,
    )
    tr = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz), bias=(z, np.ones_like(z))
    )
    # In order to get rdev = 1e-3, we need to have an a_arr, and k_arr very
    # close to the one in ClFid.
    a_arr = np.linspace(1/(1+6), 1, 38)

    tkk = ccl.halos.halomod_Tk3D_SSC_linear_bias(
        cosmo=cosmo,
        hmc=hmc,
        prof=prnfw,
        bias1=1,
        bias2=1,
        bias3=1,
        bias4=1,
        is_number_counts1=True,
        is_number_counts2=True,
        is_number_counts3=True,
        is_number_counts4=True,
    )
    # Gaussian only
    data = get_config(fsky=0.2, inc_hm=False)
    data["tracers"]["Dummy__0"]["bias"] = 1
    covcG = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    covG = covcG.get_covariance()
    _clean_tmpdir(tmpdir1)

    if sigma2B_type == "fsky":
        # Compute with fsky

        # Case 1 - fsky computed from the mask
        fsky = np.mean((mask > 0))
        sigma2_B = ccl.sigma2_B_disc(
            cosmo, a_arr=a_arr, fsky=fsky
        )  # Use 0.1 to check

        covSSC0 = ccl.angular_cl_cov_SSC(
            cosmo,
            tracer1=tr,
            tracer2=tr,
            ell=ells,
            t_of_kk_a=tkk,
            tracer3=tr,
            tracer4=tr,
            ell2=ells,
            sigma2_B=(a_arr, sigma2_B),
            integration_method="spline",
        )

        # Gaussian + non-Gaussian
        data = get_config(fsky=0.2, inc_hm=False)
        data["cov"]["SSC"] = {'compute': True, 'sigma2_B': sigma2B_type}
        data["tracers"]["Dummy__0"]["bias"] = 1
        covc1 = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
        covSSC1 = covc1.get_SSC_halomodel(0, 0, 0, 0)
        cov1 = covc1.get_covariance()
        _clean_tmpdir(tmpdir1)

        # Compare with CCL prediction
        # (interpolation errors are ~1E-4, using a very similar grid)
        assert covSSC1 == pytest.approx(covSSC0, rel=1e-4, abs=0)

        # Case 2 - fsky set by user to the "wrong" value of 0.1
        sigma2_B = ccl.sigma2_B_disc(cosmo, a_arr=a_arr, fsky=0.1)

        covSSC0 = ccl.angular_cl_cov_SSC(
            cosmo,
            tracer1=tr,
            tracer2=tr,
            ell=ells,
            t_of_kk_a=tkk,
            tracer3=tr,
            tracer4=tr,
            ell2=ells,
            sigma2_B=(a_arr, sigma2_B),
            integration_method="spline",
        )

        data = get_config(fsky=0.2, inc_hm=False)
        data["cov"]["SSC"] = {'compute': True, 'sigma2_B': sigma2B_type}
        data["cov"]["fsky_NG"] = 0.1
        data["tracers"]["Dummy__0"]["bias"] = 1
        covc1 = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
        covSSC1 = covc1.get_SSC_halomodel(0, 0, 0, 0)
        cov1 = covc1.get_covariance()
        _clean_tmpdir(tmpdir1)

        # Compare with CCL prediction
        # (interpolation errors are ~1E-4, using a very similar grid)
        assert covSSC1 == pytest.approx(covSSC0, rel=1e-4, abs=0)

    elif sigma2B_type == "mask_wl":
        # Compute with mask_wl
        area = hp.nside2pixarea(hp.npix2nside(mask.size))
        m12 = mask**2  # Same as m34
        alm = blm = hp.map2alm(m12)

        mask_wl = hp.alm2cl(alm, blm)
        mask_wl *= 2 * np.arange(mask_wl.size) + 1
        mask_wl /= np.sum(m12)**2 * area**2

        sigma2_B = ccl.sigma2_B_from_mask(cosmo, a_arr=a_arr, mask_wl=mask_wl)

        covSSC0 = ccl.angular_cl_cov_SSC(
            cosmo,
            tracer1=tr,
            tracer2=tr,
            ell=ells,
            t_of_kk_a=tkk,
            tracer3=tr,
            tracer4=tr,
            ell2=ells,
            sigma2_B=(a_arr, sigma2_B),
            integration_method="spline",
        )

        data = get_config(fsky=0.2, inc_hm=False)
        data["cov"]["SSC"] = {'compute': True, 'sigma2_B': sigma2B_type}
        data["tracers"]["Dummy__0"]["bias"] = 1
        covc1 = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
        covSSC1 = covc1.get_SSC_halomodel(0, 0, 0, 0)
        cov1 = covc1.get_covariance()
        _clean_tmpdir(tmpdir1)

        # Compare with CCL prediction
        assert covSSC1 == pytest.approx(covSSC0, rel=1e-4, abs=0)

    # Tests (using pytest.approx for more informative logs)
    # Compare result of NG method with G+SSC-G
    assert covSSC1 == pytest.approx(cov1 - covG, rel=1e-4, abs=0)


def test_file_inconsistent_errors():
    clo = get_cl_class()
    ell, cl = clo.get_ell_cl()
    # Change bpws and try to read file
    data = get_config(0.2)
    data["bpw_edges"] = data["bpw_edges"][:-1]
    data["recompute"]["cls"] = False
    data["recompute"]["mcm"] = False
    os.remove(os.path.join(tmpdir1, "data.yml"))
    clo2 = Cl(data, "Dummy__0", "Dummy__0")
    with pytest.raises(ValueError):
        clo2.get_ell_cl()
    _clean_tmpdir(tmpdir1)


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
    _clean_tmpdir(tmpdir1)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()
    _clean_tmpdir(tmpdir1)

    # Check that true Cl is within 5sigma of data Cl
    sigma = np.sqrt(np.diag(cov))

    assert np.all(np.fabs(cl_m1 - cl) < 5 * sigma)
    assert np.all(cl_class.wins == w.get_bandpower_windows())


def test_get_ell_cl_crude_error():
    cl_class = get_cl_class()
    ell, err = cl_class.get_ell_cl_crude_error()
    cl_cov = cl_class.cls_cov['cross']
    mean_mamb = cl_class.mean_mamb
    assert np.all(err == cl_class._get_cl_crude_error(
        cl_cov*mean_mamb, mean_mamb))


def test__get_cl_crude_error():
    cl_class = get_cl_class()
    # Overwrite the bin class to have more Cells per bpw
    b = nmt.NmtBin.from_nside_linear(4096, 3000)
    cl_class.b = b
    nell = 3 * 4096
    b = cl_class.b
    r0 = np.random.normal(loc=0, scale=0.1, size=nell)
    r1 = np.random.normal(loc=1, scale=0.3, size=nell)
    cl_cp = np.array([r0, r1])
    mean_mamb = 0.1
    err = cl_class._get_cl_crude_error(cl_cp, mean_mamb)

    err2 = [
        [
            np.std(r0[i * 3000 : (i + 1) * 3000]),
            np.std(r1[i * 3000 : (i + 1) * 3000]),
        ]
        for i in range(4)
    ]
    err2 = np.transpose(err2)

    assert np.abs((err * mean_mamb) / err2 - 1).max() < 1e-2


def test_get_mean_mamb():
    cl_class = get_cl_class()
    mean_mamb = cl_class.get_mean_mamb()

    m1, m2 = cl_class.get_mappers()
    ma = m1.get_mask()
    mb = m2.get_mask()

    assert np.abs(np.mean(ma * mb) / mean_mamb - 1) < 1e-5

    clfile = np.load(
        os.path.join(tmpdir1, "Dummy_Dummy", "cl_Dummy__0_Dummy__0.npz")
    )
    assert clfile["mean_mamb"] == mean_mamb


def test_custom_auto():
    off = np.pi*1E-5

    # No custom auto
    data = get_config()
    clc1 = Cl(data, "Dummy__0", "Dummy__0")
    l1, cl1 = clc1.get_ell_cl_cp()
    _clean_tmpdir(tmpdir1)

    # With custom auto
    data = get_config()
    data["tracers"]["Dummy__0"]["custom_auto"] = True
    data["tracers"]["Dummy__0"]["custom_offset"] = off
    clc2 = Cl(data, "Dummy__0", "Dummy__0")
    l2, cl2 = clc2.get_ell_cl_cp()
    _clean_tmpdir(tmpdir1)

    assert np.allclose(cl1, cl2 - off, rtol=1e-4, atol=0)

    # Covariance custom cross
    data = get_config()
    data["tracers"]["Dummy__0"]["custom_auto"] = True
    data["tracers"]["Dummy__0"]["custom_offset"] = off
    clc3 = Cl(data, "Dummy__0", "Dummy__0")
    l2, cl3 = clc3.get_ell_cl_cov()
    _clean_tmpdir(tmpdir1)

    assert np.allclose(cl1, (cl3+off)*clc3.mean_mamb, rtol=1e-4, atol=0)


def test_get_ell_cl_cp():
    # Get cl from map
    cl_class = get_cl_class()
    ell, cl = cl_class.get_ell_cl()
    ell, cl_cp = cl_class.get_ell_cl_cp()

    w = cl_class.get_workspace()
    cl2 = w.decouple_cell(cl_cp)
    _clean_tmpdir(tmpdir1)
    assert np.all(np.fabs(cl / cl2 - 1) < 1e-10)

    # Test it also in ClFid
    cl_class = get_cl_class(fiducial=True)
    ell, cl = cl_class.get_ell_cl()
    ell, cl_cp = cl_class.get_ell_cl_cp()
    cl_cp2 = w.couple_cell(cl)
    _clean_tmpdir(tmpdir1)
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
    _clean_tmpdir(tmpdir1)
    assert np.all(np.fabs(cl_binned / cl_binned2 - 1) < 1e-10)
    assert np.all(np.fabs(ell / ell_binned - 1) < 1e-10)


def test_covar_from_data():
    config = get_config(dtype0="generic")
    # Can't compute covariance unless we allow doing it from data
    with pytest.raises(NotImplementedError):
        Cov(config, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    _clean_tmpdir(tmpdir1)

    # Allow falling back to data
    config = get_config(dtype0="generic")
    config["cov"]["data_fallback"] = True
    cov_obj = Cov(config, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    cov1 = cov_obj.get_covariance()
    _clean_tmpdir(tmpdir1)

    # Compute from data on purpose
    config = get_config(dtype0="generic")
    config["cov"]["cls_from_data"] = "all"
    cov_obj = Cov(config, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    cov2 = cov_obj.get_covariance()
    _clean_tmpdir(tmpdir1)

    assert np.allclose(cov1, cov2, atol=1e-10, rtol=0)


@pytest.mark.parametrize("save_cw", [True, False])
def test_save_cw(save_cw):
    config = get_config()
    cov_class = Cov(config, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__2")
    cwname = "cw__mask_dummy0__mask_dummy0__mask_dummy0__mask_dummy2.fits"
    cwpath = os.path.join(os.path.join(config["output"], "cov"), cwname)

    # get_covariance_workspace tested through get_covariance
    cov_class.get_covariance(save_cw=save_cw)
    assert os.path.isfile(cwpath) is save_cw

    if save_cw:
        # Test that it recognizes the symmetric cases
        cov_class = Cov(config, "Dummy__0", "Dummy__0", "Dummy__2", "Dummy__0")
        cov_class.get_covariance(save_cw=save_cw)
        assert os.path.isfile(cwpath)
        cwname = "cw__mask_dummy0__mask_dummy0__mask_dummy2__mask_dummy0.fits"
        cwpath = os.path.join(os.path.join(config["output"], "cov"), cwname)
        assert not os.path.isfile(cwpath)


@pytest.mark.parametrize("cldata", ["all", "none"])
def test_get_covariance(cldata):
    # Get cl from randomnly generated map ("data")
    config = get_config(fsky=1)
    config["cov"]["cls_from_data"] = cldata

    cl_class = Cl(config, "Dummy__0", "Dummy__0")
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
    cov_class = Cov(config, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    cov = cov_class.get_covariance()

    diag = (2 * cl_m1[0] ** 2) / (2 * ell + 1) / 4
    cov_m = np.diag(diag)

    icov = np.linalg.inv(cov)
    icov_m = np.linalg.inv(cov_m)
    dCl = (cl_data - cl_m1)[0]

    chi2 = dCl.dot(icov).dot(dCl)
    chi2_m = dCl.dot(icov_m).dot(dCl)

    _clean_tmpdir(tmpdir1)
    assert np.fabs(chi2 / chi2_m - 1) < 0.03


def test_cls_vs_namaster():
    # cls
    # Get cl from randomnly generated map ("data")
    cl_class = get_cl_class()
    ell, cl_data = cl_class.get_ell_cl()
    b = cl_class.get_NmtBin()
    win = cl_class.get_bandpower_windows()
    # Read output
    clfile = np.load(
        os.path.join(tmpdir1, "Dummy_Dummy", "cl_Dummy__0_Dummy__0.npz")
    )
    _clean_tmpdir(tmpdir1)

    # Compute covariance
    cov_class = get_cov_class()
    cov = cov_class.get_covariance()
    _clean_tmpdir(tmpdir1)

    # NaMaster
    config = get_config()
    conf = config["tracers"]["Dummy__0"]
    conf["nside"] = config["sphere"]["nside"]
    conf["coords"] = config["sphere"]["coords"]
    m = MapperDummy(conf)

    # True cl
    cl_m = m.get_cl()
    spin = m.get_spin()
    mask = m.get_mask()
    signal_map = m.get_signal_map()
    n_iter_sht = config["sphere"]["n_iter_sht"]
    # Compute Cl from map
    f = nmt.NmtField(mask, signal_map, spin=spin, n_iter=n_iter_sht)
    wsp = nmt.NmtWorkspace.from_fields(f, f, b)
    cl_data_nmt_cp = nmt.compute_coupled_cell(f, f)
    cl_data_nmt = wsp.decouple_cell(cl_data_nmt_cp)

    # Couple true Cl
    cl_m_cp = wsp.couple_cell([cl_m])
    cl_m = wsp.decouple_cell(cl_m_cp)

    # Compute cov with NaMaster
    cwsp = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
    cl_cov = cl_m_cp / np.mean(mask * mask)
    cov_nmt = cwsp.gaussian_covariance(
        cl_cov, cl_cov, cl_cov, cl_cov, wsp,
        spins=[spin, spin, spin, spin])
    bpwin = wsp.get_bandpower_windows()
    icov_nmt = np.linalg.inv(cov_nmt)
    wawb = np.mean(mask**2)

    def compare(cl, cv, wn, tol=1e-5):
        rdev = cl / cl_data_nmt - 1
        assert np.max(np.abs(rdev)) < tol

        # Compare cl and covariance
        icov = np.linalg.inv(cv)
        dCl = (cl - cl_m)[0]
        chi2 = dCl.dot(icov).dot(dCl)
        chi2_m = dCl.dot(icov_nmt).dot(dCl)
        assert np.fabs(chi2 / chi2_m - 1) < tol

        # Compare bandpower windows
        assert np.max(np.abs(wn / bpwin - 1)) < tol

    compare(cl_data, cov, win)
    compare(clfile["cl"], cov, clfile["wins"])
    assert np.allclose(clfile["cl_cp"], cl_data_nmt_cp, atol=0)
    assert np.allclose(clfile["cl_cov"], cl_data_nmt_cp/wawb, atol=0)
    assert np.allclose(clfile["cl_cov_11"], cl_data_nmt_cp/wawb, atol=0)
    assert np.allclose(clfile["cl_cov_12"], cl_data_nmt_cp/wawb, atol=0)
    assert np.allclose(clfile["cl_cov_22"], cl_data_nmt_cp/wawb, atol=0)


def test_symmetric():
    data = get_config()
    # Request only 'auto' to test if read_symmetric works in the case you have
    # 'auto but you need the cross for the covariance
    data["cls"]["Dummy-Dummy"]["compute"] = "auto"
    cl_class01 = Cl(data, "Dummy__0", "Dummy__1")
    os.remove(os.path.join(tmpdir1, "data.yml"))
    cl_class10 = Cl(data, "Dummy__1", "Dummy__0")

    fname = os.path.join(cl_class10.outdir, "cl_Dummy__1_Dummy__0.npz")
    assert not os.path.isfile(fname)
    fname = os.path.join(cl_class10.outdir, "w__mask_dummy1_mask_dummy0.fits")
    assert not os.path.isfile(fname)
    assert np.all(
        np.array(cl_class01.get_masks())
        == np.array(cl_class10.get_masks()[::-1])
    )
    assert (
        np.max(
            np.abs(
                (cl_class01.get_ell_cl()[1] / cl_class10.get_ell_cl()[1] - 1)
            )
        )
        < 1e-10
    )
    assert np.all(cl_class01.get_ell_nl()[1] == cl_class10.get_ell_nl()[1])
    assert np.all(
        cl_class01.get_ell_nl_cp()[1] == cl_class10.get_ell_nl_cp()[1]
    )
    _clean_tmpdir(tmpdir1)


def test_ignore_existing_yml():
    # Test for Cls
    cl_class = get_cl_class()
    data = cl_class.data.data
    # Now, data['cls']['Dummy-Dummy']['compute'] = 'all'. We change it to
    # 'auto' and check that is read when ignore_existing_yml=True
    data["cls"]["Dummy-Dummy"]["compute"] = "auto"
    cl_class01 = Cl(data, "Dummy__0", "Dummy__1", ignore_existing_yml=True)
    assert cl_class01.data.data["cls"]["Dummy-Dummy"]["compute"] == "auto"
    cl_class01 = Cl(data, "Dummy__0", "Dummy__1", ignore_existing_yml=False)
    assert cl_class01.data.data["cls"]["Dummy-Dummy"]["compute"] == "all"

    # Test for Cov
    cov_class = Cov(
        data,
        "Dummy__0",
        "Dummy__1",
        "Dummy__0",
        "Dummy__1",
        ignore_existing_yml=True,
    )
    assert cov_class.data.data["cls"]["Dummy-Dummy"]["compute"] == "auto"
    cov_class = Cov(
        data,
        "Dummy__0",
        "Dummy__1",
        "Dummy__0",
        "Dummy__1",
        ignore_existing_yml=False,
    )
    assert cov_class.data.data["cls"]["Dummy-Dummy"]["compute"] == "all"
    _clean_tmpdir(tmpdir1)


def test_unsupported_quantity():
    data = get_config(dtype0="generic")
    with pytest.raises(NotImplementedError):
        ClFid(data, "Dummy__0", "Dummy__1")
    _clean_tmpdir(tmpdir1)


def test_symmetric_fid():
    data = get_config()
    # Request only 'auto' to test if read_symmetric works in the case you have
    # 'auto but you need the cross for the covariance
    data["cls"]["Dummy-Dummy"]["compute"] = "auto"
    cl_class01 = ClFid(data, "Dummy__0", "Dummy__1")
    os.remove(os.path.join(tmpdir1, "data.yml"))
    cl_class10 = ClFid(data, "Dummy__1", "Dummy__0")

    fname = os.path.join(cl_class10.outdir, "cl_Dummy__1_Dummy__0.npz")
    assert not os.path.isfile(fname)
    assert np.all(cl_class01.get_ell_cl()[1] == cl_class10.get_ell_cl()[1])
    _clean_tmpdir(tmpdir1)


def test_cov_nonoverlap():
    data = get_config(fsky=0.2, fsky2=0.2)
    data["tracers"]["Dummy__0"]["dec0"] = 0.0
    data["tracers"]["Dummy__1"]["dec0"] = 180.0
    covc = Cov(data, "Dummy__0", "Dummy__0", "Dummy__1", "Dummy__1")
    cov = covc.get_covariance()
    _clean_tmpdir(tmpdir1)
    assert np.all(cov == 0)


def test_cov_mmarg():
    sm = 0.1
    data = get_config(dtype0="galaxy_shear")
    data["tracers"]["Dummy__0"]["sigma_m"] = sm

    # Homemade marginalized covariance
    # First, get decoupled power spectra
    # Theory power spectra
    clf = ClFid(data, "Dummy__0", "Dummy__0")
    _, cl = clf.get_ell_cl()
    _clean_tmpdir(tmpdir1)
    # Binning
    clc = Cl(data, "Dummy__0", "Dummy__0")
    wp = clc.get_bandpower_windows()
    _clean_tmpdir(tmpdir1)
    ncl, nbpw, _, nl = wp.shape
    wp = wp.reshape((ncl * nbpw, ncl * nl))
    cl = cl.reshape(ncl * nl)
    cl = np.dot(wp, cl)
    # Marginalized covariance term
    covmargb = 4 * sm**2 * cl[:, None] * cl[None, :]

    # Do with cosmotheka (it is ordered as in NaMaster)
    covc = Cov(data, "Dummy__0", "Dummy__0", "Dummy__0", "Dummy__0")
    covmarg = covc.get_covariance_m_marg()

    covmarg = covmarg.reshape((nbpw, 4, nbpw, 4))
    covmarg00 = covmarg[:, 0, :, 0] + 1e-100
    covmargb00 = covmargb[:nbpw][:, :nbpw] + 1e-100

    assert np.amax(np.fabs(covmarg00 / covmargb00 - 1)) < 1e-5
    for i in range(4):
        for j in range(4):
            covmargij = covmarg[:, i, :, j] + 1e-100
            covmargbij = covmargb[i * nbpw : (i + 1) * nbpw][
                :, j * nbpw : (j + 1) * nbpw
            ]
            covmargbij += 1e-100

            assert np.amax(np.fabs(covmargij / covmargbij - 1)) < 1e-5

    _clean_tmpdir(tmpdir1)


@pytest.mark.parametrize(
    "perm",
    [
        [0, 0, 0, 1],  # 00, 02
        [0, 0, 1, 1],  # 00, 22
        [0, 1, 0, 0],  # 02, 00
        [0, 1, 0, 1],  # 02, 02
        [0, 1, 1, 1],  # 02, 22
        [1, 1, 0, 0],  # 22, 00
        [1, 1, 0, 1],  # 22, 02
        [1, 1, 1, 1],
    ],
)  # 22, 22
def test_cov_spin0(perm):
    nmp = [1, 2]
    nmaps = [nmp[p] for p in perm]
    ncls1 = nmaps[0] * nmaps[1]
    ncls2 = nmaps[2] * nmaps[3]
    data = get_config(
        fsky=1.0, fsky2=1.0, dtype0="galaxy_density", dtype1="galaxy_shear"
    )
    data["tracers"]["Dummy__0"]["noise_level"] = 1e-5
    data["tracers"]["Dummy__1"]["noise_level"] = 1e-10
    nbpw = len(data["bpw_edges"])

    # Spin-2 covariance class
    covc2 = Cov(
        data,
        f"Dummy__{perm[0]}",
        f"Dummy__{perm[1]}",
        f"Dummy__{perm[2]}",
        f"Dummy__{perm[3]}",
    )
    assert not covc2.spin0
    cov2 = covc2.get_covariance()
    assert cov2.shape == (ncls1 * nbpw, ncls2 * nbpw)
    _clean_tmpdir(tmpdir1)

    # Spin-0 covariance class
    data["cov"]["spin0"] = True
    covc0 = Cov(
        data,
        f"Dummy__{perm[0]}",
        f"Dummy__{perm[1]}",
        f"Dummy__{perm[2]}",
        f"Dummy__{perm[3]}",
    )
    assert covc0.spin0
    cov0 = covc0.get_covariance()
    assert cov0.shape == (ncls1 * nbpw, ncls2 * nbpw)
    _clean_tmpdir(tmpdir1)

    if ncls1 == ncls2:
        # Check that they are the same on all bandpowers
        # except the first one
        r = np.diag(cov2) / np.diag(cov0) - 1
        # This loops through EE, EB, BE, BB
        for i in range(ncls1):
            assert np.all(r[i::ncls1][1:] < 1e-5)


def test_clfid_halomod_settings():
    data = get_config()

    # Empty halo model parameters (default values)
    clf = ClFid(data, "Dummy__0", "Dummy__1")
    cosmo = clf.th.get_cosmo_ccl()
    hm_par = clf.th.get_halomodel_params()
    assert np.fabs(hm_par["mass_def"].get_Delta(cosmo, 1.0) - 200) < 1e-3
    assert hm_par["mass_def"].rho_type == "matter"
    assert hm_par["mass_func"].name == "Tinker10"
    assert hm_par["halo_bias"].name == "Tinker10"
    assert hm_par["cM"].name == "Duffy08"
    _clean_tmpdir(tmpdir1)

    # Custom halo model parameters
    md = "200c"
    mf = "Tinker08"
    hb = "Tinker10"
    cM = "Bhattacharya13"
    data["cov"]["fiducial"]["halo_model"] = {
        "mass_def": md,
        "mass_function": mf,
        "halo_bias": hb,
        "concentration": cM,
    }
    clf = ClFid(data, "Dummy__0", "Dummy__1")
    cosmo = clf.th.get_cosmo_ccl()
    hm_par = clf.th.get_halomodel_params()
    assert np.fabs(hm_par["mass_def"].get_Delta(cosmo, 1.0) - 200) < 1e-3
    assert hm_par["mass_def"].rho_type == "critical"
    assert hm_par["mass_func"].name == mf
    assert hm_par["halo_bias"].name == hb
    assert hm_par["cM"].name == cM
    _clean_tmpdir(tmpdir1)


@pytest.mark.parametrize(
    "tr1,tr2",
    [
        ("galaxy_density", "galaxy_density"),
        ("galaxy_density", "galaxy_shear"),
        ("galaxy_density", "cmb_convergence"),
        ("galaxy_shear", "galaxy_shear"),
        ("galaxy_shear", "cmb_convergence"),
        ("cmb_convergence", "cmb_convergence"),
    ],
)
def test_clfid_against_ccl(tr1, tr2):
    data = get_config(dtype0=tr1, dtype1=tr2)
    m1 = m2 = 0
    b1 = b2 = 1
    if tr1 == "galaxy_density":
        b1 = 1.1
        data["tracers"]["Dummy__0"]["bias"] = b1
        data["tracers"]["Dummy__0"]["magnif_s"] = 1
    elif tr1 == "galaxy_shear":
        m1 = 0.1
        data["tracers"]["Dummy__0"]["m"] = m1
    if tr2 == "galaxy_density":
        b2 = 1.3
        data["tracers"]["Dummy__1"]["bias"] = b2
        data["tracers"]["Dummy__1"]["magnif_s"] = 1
    elif tr2 == "galaxy_shear":
        m2 = 0.3
        data["tracers"]["Dummy__1"]["m"] = m2

    cosmo = ccl.Cosmology(**data["cov"]["fiducial"]["cosmo"])
    clf = ClFid(data, "Dummy__0", "Dummy__1")
    d = clf.get_cl_file()
    _clean_tmpdir(tmpdir1)

    def get_ccl_tracer(tr):
        if tr == "galaxy_density":
            z, nz = np.loadtxt(
                "cosmotheka/tests/data/DESY1gc_dndz_bin0.txt",
                usecols=(1, 3),
                unpack=True,
            )
            t = ccl.NumberCountsTracer(
                cosmo,
                has_rsd=False,
                dndz=(z, nz),
                bias=(z, np.ones_like(z)),
                mag_bias=(z, np.ones_like(z)),
            )
        elif tr == "galaxy_shear":
            z, nz = np.loadtxt(
                "cosmotheka/tests/data/Nz_DIR_z0.1t0.3.asc",
                usecols=(0, 1),
                unpack=True,
            )
            t = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
        elif tr == "cmb_convergence":
            t = ccl.CMBLensingTracer(cosmo, z_source=1100.0)
        return t

    t1 = get_ccl_tracer(tr1)
    t2 = get_ccl_tracer(tr2)
    factor = b1 * b2 * (1 + m1) * (1 + m2)
    clb = factor * ccl.angular_cl(cosmo, t1, t2, d["ell"])

    assert np.all(np.fabs(clb[2:] / d["cl"][0][2:] - 1) < 1e-5)


@pytest.mark.parametrize(
    "tr1,tr2",
    [
        ("galaxy_shear", "galaxy_shear"),
        ("galaxy_density", "galaxy_shear"),
        ("cmb_tSZ", "galaxy_shear"),
        ("cmb_tSZ", "cmb_convergence"),
    ],
)
def test_clfid_halomod(tr1, tr2):
    data = get_config(dtype0=tr1, dtype1=tr2, inc_hm=True)

    cosmo = ccl.Cosmology(**data["cov"]["fiducial"]["cosmo"])
    md = ccl.halos.MassDef200m
    mf = ccl.halos.MassFuncTinker10(mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mass_def=md)
    hmc = ccl.halos.HMCalculator(mass_function=mf, halo_bias=hb, mass_def=md)
    pNFW = ccl.halos.HaloProfileNFW(mass_def=md, concentration=cm)
    profs = {}
    ccltr = {}
    normed = {}
    for tr, lab in [(tr1, "Dummy__0"), (tr2, "Dummy__1")]:
        if tr == "galaxy_density":
            data["tracers"][lab]["hod_params"] = {
                "log10Mmin_0": 12.1,
                "log10M1_p": 0.1,
                "bg_0": 1.2,
            }
            profs[tr] = ccl.halos.HaloProfileHOD(
                mass_def=md,
                concentration=cm,
                log10Mmin_0=12.1,
                log10M1_p=0.1,
                bg_0=1.2,
            )
            z, nz = np.loadtxt(
                "cosmotheka/tests/data/DESY1gc_dndz_bin0.txt",
                usecols=(1, 3),
                unpack=True,
            )
            ccltr[tr] = ccl.NumberCountsTracer(
                cosmo, has_rsd=False, dndz=(z, nz), bias=(z, np.ones_like(z))
            )
            normed[tr] = True
        elif tr == "cmb_tSZ":
            data["tracers"][lab]["gnfw_params"] = {
                "mass_def": "200m",
                "mass_bias": 0.9,
            }
            profs[tr] = ccl.halos.HaloProfilePressureGNFW(
                mass_def=md, mass_bias=0.9
            )
            ccltr[tr] = ccl.tSZTracer(cosmo, z_max=3.0)
            normed[tr] = False
        elif tr == "galaxy_shear":
            profs[tr] = pNFW
            z, nz = np.loadtxt(
                "cosmotheka/tests/data/Nz_DIR_z0.1t0.3.asc",
                usecols=(0, 1),
                unpack=True,
            )
            ccltr[tr] = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
            normed[tr] = True
        elif tr == "cmb_convergence":
            profs[tr] = pNFW
            ccltr[tr] = ccl.CMBLensingTracer(cosmo, z_source=1100.0)
            normed[tr] = True

    clf = ClFid(data, "Dummy__0", "Dummy__1")
    d = clf.get_cl_file()
    _clean_tmpdir(tmpdir1)

    k_arr = np.geomspace(1e-4, 1e2, 512)
    # In order to get rdev = 1e-4, we need to have an a_arr very close to
    # the one in ClFid. Sampling on z, the second highest scale factor is a~0.8
    # instead of a~0.97, so it's better to sample on a directly.
    a_arr = np.linspace(1/(1+3), 1, 20)

    pk = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        profs[tr1],
        prof2=profs[tr2],
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
        smooth_transition=(lambda a: 0.7),
        suppress_1h=(lambda a: 0.01)
    )
    clb = ccl.angular_cl(cosmo, ccltr[tr1], ccltr[tr2], d["ell"], p_of_k_a=pk)

    assert clb[2:] == pytest.approx(d["cl"][0][2:], rel=1e-4, abs=0)


def test_clfid_halomod_M500c():
    tr1 = "cmb_tSZ"
    tr2 = "cmb_convergence"
    data = get_config(dtype0=tr1, dtype1=tr2, inc_hm=True)
    data["cov"]["fiducial"]["halo_model"] = {
        "mass_def": "500c",
        "concentration": "Duffy08M500c",
    }
    data["tracers"]["Dummy__0"]["gnfw_params"] = {
        "mass_def": "500c",
        "mass_bias": 0.9,
    }

    cosmo = ccl.Cosmology(**data["cov"]["fiducial"]["cosmo"])
    md = ccl.halos.MassDef(500, "critical")
    mf = ccl.halos.MassFuncTinker10(mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(mass_def=md)
    cm = ConcentrationDuffy08M500c(mass_def=md)
    hmc = ccl.halos.HMCalculator(mass_function=mf, halo_bias=hb, mass_def=md)
    prof1 = ccl.halos.HaloProfilePressureGNFW(mass_def=md, mass_bias=0.9)
    ccltr1 = ccl.tSZTracer(cosmo, z_max=3.0)
    prof2 = ccl.halos.HaloProfileNFW(mass_def=md, concentration=cm)
    ccltr2 = ccl.CMBLensingTracer(cosmo, z_source=1100.0)

    clf = ClFid(data, "Dummy__0", "Dummy__1")
    d = clf.get_cl_file()
    _clean_tmpdir(tmpdir1)

    k_arr = np.geomspace(1e-4, 1e2, 512)
    # In order to get rdev = 1e-4, we need to have an a_arr very close to
    # the one in ClFid. Sampling on z, the second highest scale factor is a~0.8
    # instead of a~0.97, so it's better to sample on a directly.
    a_arr = np.linspace(1/(1+6), 1, 38)

    pk = ccl.halos.halomod_Pk2D(
        cosmo,
        hmc,
        prof1,
        prof2=prof2,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
        smooth_transition=(lambda a: 0.7),
        suppress_1h=(lambda a: 0.01)
    )
    clb = ccl.angular_cl(cosmo, ccltr1, ccltr2, d["ell"], p_of_k_a=pk)

    assert clb[2:] == pytest.approx(d["cl"][0][2:], rel=1e-4, abs=0)


def test_mc_correction():
    config = get_config(dtype0="cmb_convergence", fsky=1.0, fsky2=0.1)
    cl_class = Cl(config, "Dummy__0", "Dummy__1")
    mapper_0, mapper_1 = cl_class.get_mappers()
    mask_cmbk = mapper_0.get_mask()
    _clean_tmpdir(tmpdir1)

    # Generate the sims needed to test the CMBk mc correction
    os.makedirs(tmpdir1, exist_ok=True)  # Prepare output directory
    ell = np.arange(3*NSIDE, dtype=np.float64)
    cl_input = np.zeros_like(ell)
    cl_input[2:] = 1e-3 * ell[2:]**(-3)

    nsims = 50
    # Generate the random map using synfast
    for i in range(nsims):
        # Signal and noise drawn separately
        s_map, s_alm = hp.synfast(cl_input, nside=NSIDE, alm=True)

        # Write input alm "s_alm"
        hp.write_alm(tmpdir1 + f"/sky_klm_{i:03d}.fits", s_alm)

        # To mimic the reconstruction, recover the alm after applying the mask
        # Note that we should probably account for the coupling, but for the
        # test this seems enough.
        rec_alm = hp.map2alm(s_map * mask_cmbk, lmax=3*NSIDE-1)
        hp.write_alm(tmpdir1 + f"/sim_klm_{i:03d}.fits", rec_alm)

    config = get_config(dtype0="cmb_convergence", fsky=1.0, fsky2=0.1)
    config["tracers"]["Dummy__0"]["sims_rec_path"] = tmpdir1
    config["tracers"]["Dummy__0"]["sims_in_path"] = tmpdir1
    config["cls"]["Dummy-Dummy"]["neglect_mc_correction"] = False
    cl_class = Cl(config, "Dummy__0", "Dummy__1")
    clf = cl_class.get_cl_file()

    # Check that the mc correction file was created and has the expected keys
    tl_fname = f"Tl_mask_dummy0_mask_dummy1_coordC_ns{NSIDE}.npz"
    tl_path = os.path.join(cl_class.outdir, tl_fname)
    assert os.path.isfile(tl_path)
    tl_file = np.load(tl_path)
    expected_tl_keys = {
        "Tl",
        "Tl_cp",
        "ell",
        "num",
        "denom",
        "num_cp",
        "denom_cp",
        "computed",
        "threshold"
    }
    assert set(tl_file.files) == expected_tl_keys
    assert tl_file['Tl'] == pytest.approx(np.mean(tl_file['num'], axis=0) /
                                          np.mean(tl_file['denom'], axis=0),
                                          rel=1e-10, abs=0)
    assert tl_file['Tl_cp'] == \
        pytest.approx(np.mean(tl_file['num_cp'], axis=0) /
                      np.mean(tl_file['denom_cp'], axis=0), rel=1e-10, abs=0)
    assert tl_file['computed'] == nsims

    Tl = clf['correction_cmbk']
    ell_eff = clf['ell']
    sel = ell_eff < 2 * NSIDE
    ell_eff = ell_eff[sel][1:]  # Remove first bin: noisier due to mask
    Tl = Tl[sel][1:]
    ones = np.ones_like(Tl)

    assert np.all(Tl != ones)
    # I can get rel 1e-4 in Glamdring but not in the CI, so I relax the
    # tolerance to 1e-3
    assert Tl == pytest.approx(ones, rel=1e-3, abs=0)


def test_mc_correction_resumes_partial_file(monkeypatch):
    config = get_config(dtype0="cmb_convergence", fsky=1.0, fsky2=0.1)
    config["cls"]["Dummy-Dummy"]["neglect_mc_correction"] = False
    cl_class = Cl(config, "Dummy__0", "Dummy__1")
    mapper_cmbk, mapper_x = cl_class.get_mappers()

    # Keep the test tiny: only two MC steps, one of them already stored.
    rec_sims = ["rec0", "rec1"]
    input_sims = ["in0", "in1"]
    mapper_cmbk._get_sims_fnames = lambda: (rec_sims, input_sims)
    mapper_cmbk.get_mask = lambda: np.array([1.0])
    mapper_cmbk._get_map_from_alm_file = \
        lambda path: np.array([0.0 if path.endswith("0") else 1.0])
    mapper_x.get_mask = lambda: np.array([1.0])
    mapper_x.coords = "C"

    class FakeWorkspace:
        def decouple_cell(self, cl_cp):
            return cl_cp

    cl_class.get_workspace_spin0 = lambda: FakeWorkspace()

    call_count = {"n": 0}

    def fake_nmt_field(mask, maps, spin=None):
        return {"mask": np.asarray(mask), "map": np.asarray(maps[0])}

    def fake_compute_coupled_cell(field1, field2):
        call_count["n"] += 1
        value = float(
            field1["map"][0] + 10 * field2["map"][0] + call_count["n"]
            )
        return np.array([[value]])

    monkeypatch.setattr(nmt, "NmtField", fake_nmt_field)
    monkeypatch.setattr(nmt, "compute_coupled_cell", fake_compute_coupled_cell)

    outdir = cl_class.outdir
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "Tl_mask_dummy0_mask_dummy1_coordC_ns32.npz")
    np.savez_compressed(
        fname,
        Tl=np.array([1.0]),
        Tl_cp=np.array([2.0]),
        ell=np.array([0.0]),
        num=np.array([[11.0]]),
        denom=np.array([[12.0]]),
        num_cp=np.array([[13.0]]),
        denom_cp=np.array([[14.0]]),
        computed=1,
    )

    Tl, Tl_cp = cl_class.get_correction_cmbk()

    # One precomputed step was reused, so only the missing step should be
    # evaluated here, which means two coupled-cell calls total.
    assert call_count["n"] == 2
    assert np.all(np.isfinite(Tl))
    assert np.all(np.isfinite(Tl_cp))
    saved = np.load(fname)
    assert saved["computed"] == 2
    assert saved["num"][0] == pytest.approx(11.0)
    assert saved["denom"][0] == pytest.approx(12.0)
