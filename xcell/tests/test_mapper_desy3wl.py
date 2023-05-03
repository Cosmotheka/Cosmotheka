import numpy as np
import xcell as xc
import healpy as hp
import os
import glob
import h5py
import shutil
import pytest
import fitsio


OUTDIR = 'xcell/tests/data/tmp/'
INDEX_PATH = OUTDIR  + 'desy3wl_index.h5'
NZ_PATH = OUTDIR + "nz.fits"
KINDS = ['unsheared', 'sheared_1p', 'sheared_1m', 'sheared_2p', 'sheared_2m']
COLUMNS = ['ra', 'dec', 'weight', 'e_1', 'e_2', 'psf_e1', 'psf_e2', 'R11',
           'R12', 'R21', 'R22']
ZBIN = 0
NSIDE = 32
NPIX = hp.nside2npix(32)  # 12288 pixels


def gen_index():
    # Based on save_index_short_per_bin
    def append_column(f, ds, col):
        if ds not in f:
            f.update({ds: col})

    # Unsheared columns
    columns = COLUMNS

    print(f"Creating {INDEX_PATH}")
    f = h5py.File(INDEX_PATH, mode='w')

    zbin = ZBIN
    # Selection
    # We need to save also the galaxies selected in the sheared cases for
    # the computation of Rs
    nside = NSIDE
    ra, dec = hp.pix2ang(nside, np.arange(NPIX), lonlat=True)
    position = {'ra': ra, 'dec': dec}
    numbers = np.arange(NPIX + 4000)

    # Added indices for unsheared galaxies
    ds = f'index/select_bin{zbin+1}'
    print(f"Loading {ds}", flush=True)
    append_column(f, ds, numbers[:NPIX])

    # Add indices for sheared galaxies
    for i, suffix in enumerate(["1p", "1m", "2p", "2m"]):
        ds = f'index/select_{suffix}_bin{zbin+1}'
        print(f"Loading {ds}", flush=True)
        append_column(f, ds, numbers[NPIX + i*1000: NPIX + (i+1)*1000])

    # Columns
    # Add unsheared columns
    for i, col in enumerate(columns):
        ds = f"catalog/metacal/unsheared/{col}"
        print(f"Loading {ds}", flush=True)
        if col in ['ra', 'dec']:
            col = position[col]
        else:
            col = np.ones(numbers.size) * i
        append_column(f, ds, col)

    # Add sheared galaxies columns
    for i, grp in enumerate(['sheared_1p', 'sheared_1m', 'sheared_2p',
                             'sheared_2m']):
        i += 1  # To be able to use COLUMNS.index(blabla)
        # Columns
        for col in ['weight', 'e_1', 'e_2']:
            ds = f"catalog/metacal/{grp}/{col}"
            print(f"Loading {ds}", flush=True)
            col = np.ones(numbers.size) * (columns.index(col) + i)
            append_column(f, ds, col)
    f.close()

def gen_nz():
    fits = fitsio.read('xcell/tests/data/cat_zbin.fits')
    with fitsio.FITS(NZ_PATH, "rw") as f:
        f.write(fits, extname="nz_source")

def remove_rerun(prerun):
    frerun = glob.glob(prerun + 'DESY3wl*.fits*')
    for f in frerun:
        os.remove(f)
    fn = prerun + 'mask_mask_coordC_ns32.fits.gz'
    if os.path.isfile(fn):
        os.remove(fn)


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)
    gen_index()
    gen_nz()


def teardown_module():
    shutil.rmtree(OUTDIR, ignore_errors=True)


@pytest.fixture
def config():
    return {'indexcat': INDEX_PATH,
            'file_nz': NZ_PATH,
            'mode': 'shear',
            'zbin': ZBIN, 'nside': NSIDE, 'mask_name': 'mask',
            'coords': 'C'}


@pytest.fixture
def mapper(config):
    return xc.mappers.MapperDESY3wl(config)


def test_smoke(config):
    # Checks for errors in summoning mapper
    xc.mappers.MapperDESY3wl(config)


def test_check_kind(mapper):
    for k in KINDS:
        mapper._check_kind(k)

    with pytest.raises(ValueError):
        mapper._check_kind('fails')


def test_get_cat_index(mapper):
    c2 = mapper._get_cat_index()
    assert isinstance(c2, h5py.File)


@pytest.mark.parametrize("kind", KINDS)
def test_get_select(kind, mapper):
    sel = mapper._get_select(kind)
    numbers = np.arange(NPIX + 4000)
    if kind == "unsheared":
        assert np.all(sel == numbers[:NPIX])
    else:
        i = KINDS.index(kind) - 1
        assert np.all(sel == numbers[NPIX+i*1000: NPIX+(i+1)*1000])


@pytest.mark.parametrize("kind", KINDS)
def test_get_ellips(mapper, kind):
    ellips = mapper._get_ellips(kind)
    numbers = np.arange(NPIX)
    c = COLUMNS.index('e_1')
    if kind == "unsheared":
        assert ellips.shape == (2, NPIX)
    else:
        i = KINDS.index(kind)
        c += i
        assert ellips.shape == (2, 1000)
    assert np.all(ellips == np.array((c, c+1))[:, None])


def test_get_ellips_unbiased(mapper):
    # For constant ellipticities the sh map should be 0 as we remove the mean
    # ellipticity of the set
    e_unb = mapper.get_ellips_unbiased()
    assert np.all(e_unb == 0)


def test_get_positions(mapper):
    positions = mapper.get_positions()
    ra, dec = hp.pix2ang(NSIDE, np.arange(NPIX), lonlat=True)
    assert isinstance(positions, dict)
    assert positions['ra'].size == NPIX
    assert positions['dec'].size == NPIX
    assert np.all(positions['ra'] == ra)
    assert np.all(positions['dec'] == dec)


@pytest.mark.parametrize("kind", KINDS)
def test_get_weights(kind, mapper):
    weights = mapper.get_weights(kind)
    c = COLUMNS.index('weight')
    if kind == "unsheared":
        assert weights.size == NPIX
    else:
        assert weights.size == 1000
    i = KINDS.index(kind)
    assert np.all(weights == c + i)


@pytest.mark.parametrize("mode", ["PSF", "shear", "fail"])
def test_set_mode(mapper, mode):
    if mode == "PSF":
        e1f, e2f, m = mapper._set_mode(mode)
        assert e1f == "psf_e1"
        assert e2f == "psf_e2"
        assert m == "PSF"
    elif mode == "shear":
        e1f, e2f, m = mapper._set_mode(mode)
        assert e1f == "e_1"
        assert e2f == "e_2"
        assert m == "shear"
    else:
        with pytest.raises(ValueError):
            e1f, e2f, m = mapper._set_mode(mode)


def test_get_Rs(mapper):
    # Rs is 0 because the columns have constant values, so the selection will
    # not matter.
    assert np.all(mapper._get_Rs() == 0)


def test_remove_multiplicative_bias(mapper):
    e = mapper._get_ellips()
    e_unb = mapper._remove_multiplicative_bias(e)
    Rg = 0.5 * (COLUMNS.index('R11') +  COLUMNS.index('R22'))
    # Rs = 0
    assert np.all(e_unb == e/Rg)


def test_get_ellipticities_maps(mapper):
    sh = np.array(mapper._get_ellipticity_maps('shear'))
    psf = np.array(mapper._get_ellipticity_maps('PSF'))
    mask = mapper.get_mask()
    goodpix = mask > 0

    assert sh.shape == (2, hp.nside2npix(32))
    assert psf.shape == (2, hp.nside2npix(32))
    # for constant ellipticities the sh map should be 0
    # as we remove the mean ellipticity of the set
    assert np.all(np.fabs(sh) < 1E-5)
    psf_e1 = COLUMNS.index('psf_e1')
    psf_e2 = COLUMNS.index('psf_e2')
    assert np.all((np.fabs(psf[0])-psf_e1)[goodpix] < 1E-5)
    assert np.all((np.fabs(psf[1])-psf_e2)[goodpix] < 1E-5)


@pytest.mark.parametrize("mode", ["PSF", "shear"])
def test_get_signal_map(mapper, mode):
    m = mapper.get_signal_map(mode)
    assert np.all(m == mapper._get_ellipticity_maps(mode))


def test_get_nz(mapper):
    z, nz = mapper.get_nz()
    assert np.all(z == 0.6 * np.ones(mapper.npix))
    assert np.all(nz == (mapper.zbin + 1) * np.ones(mapper.npix))


def test_get_mask(mapper):
    mask = mapper.get_mask()
    assert len(mask) == hp.nside2npix(32)
    # One item per pix
    goodpix = mask > 0
    mask = mask[goodpix]
    assert np.mean(mask) == COLUMNS.index('weight')


def test_get_nl_coupled(mapper):
    # TODO: Update this
    mask = mapper.get_mask()
    goodpix = mask > 0
    fsk = len(mask[goodpix])/len(mask)
    aa = fsk*hp.nside2pixarea(32)
    sh = mapper.get_nl_coupled()
    assert np.all(sh[0] == 0)

    psf = mapper.get_nl_coupled('PSF')
    w = COLUMNS.index('weight')
    psfp = (COLUMNS.index('psf_e1')**2 + COLUMNS.index('psf_e2')**2) * w**2 / 2
    psfp *= aa
    assert np.all(psf[0][:2] == 0)
    assert np.fabs(np.mean(psf[0][2:])-psfp) < 1E-5


def test_get_dtype(mapper):
    dtype = mapper.get_dtype()
    assert dtype == 'galaxy_shear'


def test_get_spin(mapper):
    assert mapper.get_spin() == 2


def test_rerun(config):
    config['path_rerun'] = OUTDIR
    remove_rerun(OUTDIR)
    # Initialize mapper
    m = xc.mappers.MapperDESY3wl(config)
    # Get maps and catalog
    s = m.get_signal_map()
    psf = m.get_signal_map('PSF')
    mask = m.get_mask()
    nl_cp = m.get_nl_coupled()
    nl_cp_psf = m.get_nl_coupled('PSF')

    # Check rerun files have been created
    zbin = config['zbin']
    nside = config['nside']


    for fname in [f'signal_map_shear_coordC_ns{nside}.fits.gz',
                  f'signal_map_PSF_coordC_ns{nside}.fits.gz',
                  f'shear_w2s2_coordC_ns{nside}.fits.gz',
                  f'PSF_w2s2_coordC_ns{nside}.fits.gz']:
        assert os.path.isfile(os.path.join(OUTDIR,
                                           f"DESY3wl_bin{zbin}_{fname}"))
    assert os.path.isfile(os.path.join(OUTDIR,
                                       f'mask_mask_coordC_ns{nside}.fits.gz'))

    # Check we recover the same mask and catalog
    # Non-exsisting fits files - read from rerun
    config['data_cat'] = 'whatever'
    m_rerun = xc.mappers.MapperDESY3wl(config)
    s_from_rerun = m_rerun.get_signal_map()
    psf_from_rerun = m_rerun.get_signal_map('PSF')
    mask_from_rerun = m_rerun.get_mask()
    nl_cp_from_rerun = m_rerun.get_nl_coupled()
    nl_cp_psf_from_rerun = m_rerun.get_nl_coupled('PSF')
    assert np.all(np.array(s) == np.array(s_from_rerun))
    assert np.all(np.array(psf) == np.array(psf_from_rerun))
    assert np.all(mask == mask_from_rerun)
    assert np.all(nl_cp == nl_cp_from_rerun)
    assert np.all(nl_cp_psf == nl_cp_psf_from_rerun)

    # Clean rerun
    remove_rerun(OUTDIR)
