import os
import pytest
import numpy as np
import sacc
import shutil
from xcell.cls.to_sacc import sfile
from xcell.cls.data import Data
from xcell.cls.cl import Cl, ClFid

# Remove previous test results
tmpdir = './xcell/tests/cls/dummy1'


def remove_tmpdir():
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)


def get_config(fsky0=0.2, fsky1=0.3, dtype0='galaxy_density',
               dtype1='galaxy_density'):
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
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky0, 'seed': 0,
              'dtype': dtype0}
    dummy1 = {'mask_name': 'mask_dummy1', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky1, 'seed': 100,
              'dtype': dtype1}
    bpw_edges = list(range(0, 3 * nside, 4))

    return {'tracers': {'Dummy__0': dummy0, 'Dummy__1': dummy1},
            'cls': {'Dummy-Dummy': {'compute': 'all'}},
            'cov': {'fiducial': {'cosmo': cosmo, 'gc_bias':  False, 'wl_m':
                                 False, 'wl_ia': False}},
            'bpw_edges': bpw_edges,
            'healpy': {'n_iter_sht': 0, 'n_iter_mcm': 3, 'n_iter_cmcm': 3,
                       'nside': nside},
            'recompute': {'cls': False, 'cov': False, 'mcm': False, 'cmcm':
                          False},
            'output': tmpdir}


def get_data(fsky=0.2, fsky2=0.3, ):
    config = get_config(fsky, fsky2)
    return Data(data=config)


def get_sfile(use='cls', m_marg=False):
    # Generate data.yml file
    data = get_data()
    datafile = os.path.join(data.data['output'], 'data.yml')
    return sfile(datafile, 'cls_cov_dummy.fits', use, m_marg)


@pytest.mark.parametrize('use', ['cls', 'nl', 'fiducial', 'potato'])
def test_init(use):
    outdir = get_config()['output']
    if use != 'potato':
        get_sfile(use)
        sfile_path = os.path.join(outdir, 'cls_cov_dummy.fits')
        assert os.path.isfile(sfile_path)
    else:
        with pytest.raises(ValueError):
            get_sfile(use)


def test_added_tracers():
    s = get_sfile()
    data = get_data()
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
def test_ell_cl():
    s = get_sfile(use)

if os.path.isdir(tmpdir):
    shutil.rmtree(tmpdir)
