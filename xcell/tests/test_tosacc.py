import os
import pytest
import numpy as np
import sacc
import shutil
from xcell.cls.to_sacc import ClSack
from xcell.cls.data import Data
from xcell.cls.cl import Cl, ClFid
from xcell.cls.cov import Cov

# Remove previous test results
tmpdir = './xcell/tests/cls/dummy1'


def remove_tmpdir():
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)


def get_config(fsky0=0.2, fsky1=0.3, dtype0='galaxy_density',
               dtype1='galaxy_shear'):
    nside = 32
    # Set only the necessary entries. Leave the others to their default
    # value.
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
              'cosmo': cosmo, 'fsky': fsky0, 'seed': 0,
              'dtype': dtype0}
    dummy1 = {'mask_name': 'mask_dummy1', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'fsky': fsky1, 'seed': 100,
              'dtype': dtype1}
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0, 'Dummy__1': dummy1},
            'cls': {'Dummy-Dummy': {'compute': 'all'}},
            'cov': {'fiducial': {'cosmo': cosmo, 'wl_m':
                                 False, 'wl_ia': False}},
            'bpw_edges': bpw_edges,
            'sphere': {'n_iter_sht': 0, 'n_iter_mcm': 3, 'n_iter_cmcm': 3,
                       'nside': nside, 'coords': 'C'},
            'recompute': {'cls': False, 'cov': False, 'mcm': False, 'cmcm':
                          False},
            'output': tmpdir}


def get_data(fsky0=0.2, fsky1=0.3, dtype0='galaxy_density',
             dtype1='galaxy_shear'):
    config = get_config(fsky0, fsky1, dtype0, dtype1)
    return Data(data=config)


def get_ClSack(use='cls', m_marg=False, fsky0=0.2, fsky1=0.3,
               dtype0='galaxy_density', dtype1='galaxy_shear'):
    # Generate data.yml file
    data = get_data(dtype0=dtype0, dtype1=dtype1)
    datafile = os.path.join(data.data['output'], 'data.yml')
    return ClSack(datafile, 'cls_cov_dummy.fits', use, m_marg)


@pytest.mark.parametrize('use', ['cls', 'nl', 'fiducial', 'potato'])
def test_init(use):
    outdir = get_config()['output']
    if use != 'potato':
        get_ClSack(use)
        ClSack_path = os.path.join(outdir, 'cls_cov_dummy.fits')
        assert os.path.isfile(ClSack_path)
    else:
        with pytest.raises(ValueError):
            get_ClSack(use)


@pytest.mark.parametrize('dt1, dt2', [('galaxy_density', 'galaxy_shear'),
                                      ('galaxy_shear', 'cmb_convergence'),
                                      ('galaxy_density', 'cmb_convergence')])
def test_added_tracers(dt1, dt2):
    s = get_ClSack(dtype0=dt1, dtype1=dt2)
    data = get_data(dtype0=dt1, dtype1=dt2)
    for trname in data.data['tracers'].keys():
        tr = s.s.tracers[trname]
        m = data.get_mapper(trname)
        assert tr.quantity == m.get_dtype()
        if tr.quantity in ['galaxy_density', 'galaxy_shear']:
            assert isinstance(tr, sacc.tracers.NZTracer)
            z, nz = m.get_nz()
            assert np.all(tr.z == z)
            assert np.all(tr.nz == nz)
        elif tr.quantity in ['cmb_convergence']:
            assert isinstance(tr, sacc.tracers.MapTracer)
            assert tr.ell == m.get_ell()
            # Only here because tr.spin is not an attribute of NZTracers
            assert tr.spin == m.get_spin()
            assert tr.beam_extra['nl'] == m.get_nl_coupled()[0]
            assert tr.beam == np.ones_like(tr.ell)
        else:
            raise ValueError('Tracer not implemented')


@pytest.mark.parametrize('use', ['cls', 'nl', 'fiducial'])
def test_ell_cl_autocov(use):
    s = get_ClSack(use)
    data = get_data()

    for dtype in s.s.get_data_types():
        for trs in s.s.get_tracer_combinations(dtype):
            if dtype in ['cl_00', 'cl_0e', 'cl_ee']:
                ix = 0
            elif dtype in ['cl_0b', 'cl_eb']:
                ix = 1
            elif dtype in ['cl_be']:
                ix = 2
            elif dtype in ['cl_bb']:
                ix = 3
            else:
                raise ValueError(f'data_type {dtype} must be weird!')

            if use == 'fiducial':
                cl_class = ClFid(data.data, *trs)
            else:
                cl_class = Cl(data.data, *trs)

            if use == 'nl':
                ell, cl = s.s.get_ell_cl(dtype, trs[0], trs[1])
                ell2, cl2 = cl_class.get_ell_nl()
            else:
                ell, cl, cov = s.s.get_ell_cl(dtype, trs[0], trs[1],
                                              return_cov=True)
                ell2, cl2 = cl_class.get_ell_cl()

                cov_class = Cov(data.data, *trs, *trs)
                nbpw = ell.size
                ncls = cl2.shape[0]
                cov2 = cov_class.get_covariance()
                cov2 = cov2.reshape((nbpw, ncls, nbpw, ncls))[:, ix, :, ix]
                assert np.max(np.abs((cov - cov2) / np.mean(cov))) < 1e-5

            if use == 'fiducial':
                # Matrices to bin the fiducial Cell
                ws_bpw = np.zeros((ell.size, ell2.size))
                ws_bpw[np.arange(ell.size), ell.astype(int)] = 1
                assert np.all(cl == ws_bpw.dot(cl2[ix]))
            else:
                assert np.all(cl == cl2[ix])
                assert np.all(ell == ell2)


def test_get_dof_tracers():
    s = get_ClSack()
    for tr1, tr2 in s.s.get_tracer_combinations():
        s1 = np.max((s.data.get_mapper(tr1).get_spin(), 1))
        s2 = np.max((s.data.get_mapper(tr2).get_spin(), 1))

        dof = s1 * s2
        assert dof == s.get_dof_tracers((tr1, tr2))


def test_get_datatypes_from_dof():
    s = get_ClSack()
    assert s.get_datatypes_from_dof(1) == ['cl_00']
    assert s.get_datatypes_from_dof(2) == ['cl_0e', 'cl_0b']
    assert s.get_datatypes_from_dof(4) == ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
    with pytest.raises(ValueError):
        s.get_datatypes_from_dof(3)


# Just test with covG to speed up the tests. The covariance is built in the
# same way.
@pytest.mark.parametrize('m_marg', [False])
def test_covariance_G(m_marg):
    s = get_ClSack(m_marg=m_marg)

    nbpw = len(s.data.data['bpw_edges'])

    ix_cov_d = {'cl_00': 0, 'cl_0e': 0, 'cl_0b': 1, 'cl_ee': 0, 'cl_eb': 1,
                'cl_be': 2, 'cl_bb': 3}

    for dt1 in s.s.get_data_types():
        for trs1 in s.s.get_tracer_combinations(data_type=dt1):
            dof1 = s.get_dof_tracers(trs1)
            ix_cov1 = ix_cov_d[dt1]
            ix1 = s.s.indices(data_type=dt1, tracers=trs1)
            for dt2 in s.s.get_data_types():
                for trs2 in s.s.get_tracer_combinations(data_type=dt2):
                    dof2 = s.get_dof_tracers(trs2)
                    ix_cov2 = ix_cov_d[dt2]
                    ix2 = s.s.indices(data_type=dt2, tracers=trs2)

                    if m_marg:
                        cov = Cov(s.data.data, *trs1,
                                  *trs2).get_covariance_m_marg()
                    else:
                        cov = Cov(s.data.data, *trs1, *trs2).get_covariance()
                    cov = cov.reshape((nbpw, dof1, nbpw, dof2))
                    cov = cov[:, ix_cov1, :, ix_cov2]

                    scov = s.s.covariance.covmat[ix1][:, ix2]
                    if np.any(cov):
                        assert np.max(np.abs((scov - cov)
                                             / np.mean(cov))) < 1e-5
                    else:
                        assert ~np.any(scov)


def test_covariance_extra():
    # Generate a config file with extra covariance
    config = get_config().copy()
    config['cov'].update({'extra': {'path': os.path.join(tmpdir,
                                                         'dummy_cov.npy'),
                                    'order': ['Dummy-Dummy',
                                              'Dummy-DummyWL',
                                              'Dummy-DummyCV',
                                              'DummyWL-DummyWL',
                                              'DummyWL-DummyCV',
                                              'DummyCV-DummyCV']}})

    config['cls'].update({'Dummy-Dummy': {'compute': 'all'},
                          'Dummy-DummyWL': {'compute': 'all'},
                          'Dummy-DummyCV': {'compute': 'all'},
                          'DummyWL-DummyWL': {'compute': 'auto'},
                          'DummyWL-DummyCV': {'compute': 'all'},
                          'DummyCV-DummyCV': {'compute': 'all'}})

    # Add extra difficulty by adding 2 more tracers
    config['tracers']['DummyWL'] = config['tracers']['Dummy__1'].copy()
    config['tracers']['DummyCV'] = config['tracers']['Dummy__0'].copy()
    config['tracers']['DummyCV']['dtype'] = 'cmb_convergence'

    # Overwrite 'data.yml' with new configuration
    data = Data(data=config, override=True)
    datafile = os.path.join(data.data['output'], 'data.yml')

    # Populate a sacc file with cls and covG. This will be used as extra
    # covariance later. Note that the tracers are in different order than in
    # the config file when reading the yml file.
    s = ClSack(datafile, 'cls_cov_dummy.fits', 'cls')

    covmat = s.s.covariance.covmat
    # Set B-modes to 0 as they are set to 0 when reading the extra covariance
    for dt in s.s.get_data_types():
        if 'b' in dt:
            ix = s.s.indices(data_type=dt)
            covmat[ix] = 0
            covmat[:, ix] = 0

    # Prepare the "extra" covariance. So far extra covs with B-modes are not
    # implemented Keep only spin-0 and E-modes
    # Not done in previous loop because the indices vary for the sacc file and
    # not longer correspond to those in covmat
    for dt in s.s.get_data_types():
        if 'b' in dt:
            s.s.remove_selection(data_type=dt)

    # Reorder the covariance as stated in the config
    ix_reorder_d = {k: [] for k in s.data.data['cls'].keys()}
    for trs in s.s.get_tracer_combinations():
        key = s.data.get_tracers_bare_name_pair(*trs)
        ix_reorder_d[key].extend(s.s.indices(tracers=trs))

    ix_reorder = []
    for key in s.data.data['cov']['extra']['order']:
        if key not in ix_reorder_d:
            key = '-'.join(key.split('-')[::-1])
        ix_reorder.extend(ix_reorder_d[key])

    # Save the reordered covariance
    np.save(config['cov']['extra']['path'],
            s.s.covariance.covmat[ix_reorder][:, ix_reorder])

    # Populate a sacc file with nls and cov extra (read previous cov)
    s2 = ClSack(datafile, 'cls_cov_dummy.fits', 'nl')
    covmat2 = s2.s.covariance.covmat

    # Remove B-modes to test if spin-0 and E-modes are correctly taken into
    # account
    for dt in s2.s.get_data_types():
        if 'b' in dt:
            s2.s.remove_selection(data_type=dt)

    # Check if the covariance is correctly read with no B-modes
    cov_no_B = s.s.covariance.covmat
    cov_no_B2 = s2.s.covariance.covmat
    assert np.max(np.abs((cov_no_B - cov_no_B2)/np.mean(cov_no_B))) < 1e-5
    # Check if the full covariance is correctly generated
    assert np.max(np.abs((covmat2 - covmat)/np.mean(covmat))) < 1e-5


remove_tmpdir()
