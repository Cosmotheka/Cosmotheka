#!/usr/bin/python
from xcell.cls import tools
import pymaster as nmt
import healpy as hp
import numpy as np
import os
import pytest

tmpdir = 'xcell/tests/cls/'
dummyfile = tmpdir + 'dummyfile.fits'


def get_wsp(cwsp=False):
    b = nmt.NmtBin(nside=32, nlb=10)

    mask = hp.read_map('xcell/tests/data/mask1.fits')
    f = nmt.NmtField(mask, [mask], spin=0, n_iter=0)

    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, bins=b, n_iter=0)

    if not cwsp:
        return w
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(f, f)

    return cw


def create_corrupted_file():
    with open(dummyfile, 'w') as f:
        f.write("kjkdjkfjdkjkjfljsdjfkljsdkafjkldjlk")


# Cleaning the tmp dir before running and after running the tests
@pytest.fixture(autouse=True)
def run_clean_tmp():
    if os.path.isfile(dummyfile):
        os.remove(dummyfile)


def test_save_npz():
    fname = os.path.join(tmpdir, 'test.npz')
    ell = np.arange(10)
    cl = np.ones(10)
    tools.save_npz(fname, ell=ell, cl=cl)
    f = np.load(fname)
    assert np.all(ell == f['ell'])
    assert np.all(cl == f['cl'])

    with pytest.raises(RuntimeError):
        tools.save_npz(fname, ell=ell, cl=cl, fail=[np.nan])

    with pytest.raises(RuntimeError):
        tools.save_npz(fname, ell=ell, cl=cl, fail=[1e128])


@pytest.mark.parametrize('cwsp', [False, True])
def test_save_wsp(cwsp):
    w = get_wsp(cwsp)
    # Check that it correctly saves the file
    tools.save_wsp(w, dummyfile)
    assert os.path.isfile(dummyfile)

    w2 = nmt.NmtWorkspace() if not cwsp else nmt.NmtCovarianceWorkspace()
    w2.read_from(dummyfile)

    if not cwsp:
        mcm = w.get_coupling_matrix() + 1e-100
        mcm2 = w2.get_coupling_matrix() + 1e-100
        assert np.max(np.abs((mcm / mcm2 - 1))) < 1e-5
    else:
        assert w.wsp.lmax == w2.wsp.lmax

    # Check that it raises an error if it fails writing the file but it doesn't
    # exist
    with pytest.raises(RuntimeError):
        tools.save_wsp(w, 'unexsitentfolder/dummyfile.fits')

    # TODO: We need to test that if it fails to save the file, it removes the
    # corrupted file and save it again.
    os.remove(dummyfile)


@pytest.mark.parametrize('cwsp', [False, True])
def test_read_wsp(cwsp):
    w = get_wsp(cwsp)
    w.write_to(dummyfile)

    # Check it reads the file correctly
    w2 = nmt.NmtWorkspace() if not cwsp else nmt.NmtCovarianceWorkspace()
    tools.read_wsp(w2, dummyfile)

    if not cwsp:
        mcm = w.get_coupling_matrix() + 1e-100
        mcm2 = w2.get_coupling_matrix() + 1e-100
        assert np.max(np.abs((mcm / mcm2 - 1))) < 1e-5
    else:
        assert w.wsp.lmax == w2.wsp.lmax

    # Check you can pass kwargs
    if not cwsp:
        tools.read_wsp(w2, dummyfile, read_unbinned_MCM=False)
        assert not w2.has_unbinned
    else:
        with pytest.raises(TypeError):
            tools.read_wsp(w2, dummyfile, read_unbinned_MCM=False)
        tools.read_wsp(w2, dummyfile, force_spin0_only=True)
        assert w2.wsp.spin0_only == 1

    # Check read_wsp removes the file if it fails to read it
    create_corrupted_file()
    tools.read_wsp(w2, dummyfile)
    assert not os.path.isfile(dummyfile)

    # TODO: We need to test that it raises an error if it fails while reading
    # the file but not with RuntimeError
