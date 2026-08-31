#!/usr/bin/python
from cosmotheka.cls import tools
import pymaster as nmt
import healpy as hp
import numpy as np
import os
import pytest

tmpdir = 'cosmotheka/tests/cls/'
dummyfile = tmpdir + 'dummyfile.fits'


def get_wsp(cwsp=False):
    b = nmt.NmtBin.from_nside_linear(nside=32, nlb=10)

    mask = hp.read_map('cosmotheka/tests/data/mask1.fits')
    f = nmt.NmtField(mask, [mask], spin=0, n_iter=0)

    w = nmt.NmtWorkspace.from_fields(f, f, b)

    if not cwsp:
        return w
    cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)

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

    # Threshold
    tools.save_npz(fname, threshold=1e10, ell=ell, cl=cl, fail=[1e9])
    with pytest.raises(RuntimeError):
        tools.save_npz(fname, threshold=1e10, ell=ell, cl=cl, fail=[1e11])


@pytest.mark.parametrize('cwsp', [False, True])
def test_save_wsp(cwsp):
    w = get_wsp(cwsp)
    # Check that it correctly saves the file
    tools.save_wsp(w, dummyfile)
    assert os.path.isfile(dummyfile)

    if cwsp:
        w2 = nmt.NmtCovarianceWorkspace.from_file(dummyfile)
    else:
        w2 = nmt.NmtWorkspace.from_file(dummyfile)

    if not cwsp:
        mcm = w.get_coupling_matrix() + 1e-100
        mcm2 = w2.get_coupling_matrix() + 1e-100
        assert np.max(np.abs((mcm / mcm2 - 1))) < 1e-5
    else:
        assert w.wsp.lmax == w2.wsp.lmax

    # Check that it raises an error if it fails writing the file but it doesn't
    # exist
    err = OSError if cwsp else RuntimeError
    with pytest.raises(err):
        tools.save_wsp(w, 'unexsitentfolder/dummyfile.fits')

    # TODO: We need to test that if it fails to save the file, it removes the
    # corrupted file and save it again.
    os.remove(dummyfile)


@pytest.mark.parametrize('cwsp', [False, True])
def test_read_wsp(cwsp):
    w = get_wsp(cwsp)
    w.write_to(dummyfile)

    # Check it reads the file correctly
    w2 = tools.read_wsp(dummyfile, cwsp)

    if not cwsp:
        mcm = w.get_coupling_matrix() + 1e-100
        mcm2 = w2.get_coupling_matrix() + 1e-100
        assert np.max(np.abs((mcm / mcm2 - 1))) < 1e-5
    else:
        assert w.wsp.lmax == w2.wsp.lmax

    # Check read_wsp removes the file if it fails to read it
    create_corrupted_file()
    tools.read_wsp(dummyfile, cwsp)
    assert not os.path.isfile(dummyfile)

    # TODO: We need to test that it raises an error if it fails while reading
    # the file but not with RuntimeError
