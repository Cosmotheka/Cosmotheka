import numpy as np
import xcell as xc
import healpy as hp
import os
import glob
from astropy.table import Table, hstack


def get_config():
    return {'data_cat': 'xcell/tests/data/catalog.fits',
            'zbin_cat': 'xcell/tests/data/cat_zbin.fits',
            'file_nz': 'xcell/tests/data/cat_zbin.fits',
            'zbin': 1, 'nside': 32, 'mask_name': 'mask',
            'coords': 'C'}


def get_mapper():
    return xc.mappers.MapperDESY1wl(get_config())


def remove_rerun(prerun):
    frerun = glob.glob(prerun + 'DESY1wl*.fits*')
    for f in frerun:
        os.remove(f)
    fn = prerun + 'mask_mask_coordC_ns32.fits.gz'
    if os.path.isfile(fn):
        os.remove(fn)


def test_smoke():
    # Checks for errors in summoning mapper
    get_mapper()


def get_es():
    # calculates an estimate for the signal of a periodic
    # set of ellipticities in the sky
    # (see test_get_signal_map())
    npix = hp.nside2npix(32)
    return np.repeat(np.array([np.arange(4)]), npix//4,
                     axis=0).flatten()


def test_load_catalog():
    config = get_config()
    m = get_mapper()
    cat = m._load_catalog()

    columns_data = ['coadd_objects_id', 'e1', 'e2',
                    'psf_e1', 'psf_e2', 'ra', 'dec',
                    'R11', 'R12', 'R21', 'R22',
                    'flags_select']
    columns_zbin = ['zbin_mcal', 'zbin_mcal_1p',
                    'zbin_mcal_1m', 'zbin_mcal_2p', 'zbin_mcal_2m']

    cat_data = Table.read(config['data_cat'], format='fits')
    cat_data.keep_columns(columns_data)

    cat_zbin = Table.read(config['zbin_cat'], format='fits')
    cat_zbin.keep_columns(columns_zbin)
    cat_data = hstack([cat_data, cat_zbin])

    cat_data.remove_rows(cat_data['dec'] < -90)
    cat_data.remove_rows(cat_data['dec'] > -35)

    assert(len(cat_data) == len(cat))
    assert(np.all(cat_data == cat))


def test_rerun():
    config = get_config()
    prerun = 'xcell/tests/data/'
    config['path_rerun'] = prerun
    remove_rerun(prerun)
    # Initialize mapper
    m = xc.mappers.MapperDESY1wl(config)
    # Get maps and catalog
    cat = m.get_catalog()
    s = m.get_signal_map()
    psf = m.get_signal_map('PSF')
    mask = m.get_mask()
    nl_cp = m.get_nl_coupled()
    nl_cp_psf = m.get_nl_coupled('PSF')

    # Check rerun files have been created
    zbin = config['zbin']
    nside = config['nside']

    for fname in [f'catalog_rerun_bin{zbin}.fits',
                  f'signal_map_shear_bin{zbin}_coordC_ns{nside}.fits.gz',
                  f'signal_map_PSF_bin{zbin}_coordC_ns{nside}.fits.gz',
                  f'shear_w2s2_bin{zbin}_coordC_ns{nside}.fits.gz',
                  f'PSF_w2s2_bin{zbin}_coordC_ns{nside}.fits.gz']:
        assert os.path.isfile(os.path.join(prerun, "DESY1wl_" + fname))
    assert os.path.isfile(os.path.join(prerun,
                                       f'mask_mask_coordC_ns{nside}.fits.gz'))

    # Check we recover the same mask and catalog
    # Non-exsisting fits files - read from rerun
    config['data_cat'] = 'whatever'
    m_rerun = xc.mappers.MapperDESY1wl(config)
    cat_from_rerun = m_rerun.get_catalog()
    s_from_rerun = m_rerun.get_signal_map()
    psf_from_rerun = m_rerun.get_signal_map('PSF')
    mask_from_rerun = m_rerun.get_mask()
    nl_cp_from_rerun = m_rerun.get_nl_coupled()
    nl_cp_psf_from_rerun = m_rerun.get_nl_coupled('PSF')
    assert np.all(cat == cat_from_rerun)
    assert np.all(np.array(s) == np.array(s_from_rerun))
    assert np.all(np.array(psf) == np.array(psf_from_rerun))
    assert np.all(mask == mask_from_rerun)
    assert np.all(nl_cp == nl_cp_from_rerun)
    assert np.all(nl_cp_psf == nl_cp_psf_from_rerun)

    # Clean rerun
    remove_rerun(prerun)


def test_get_signal_map():
    m = get_mapper()
    sh = np.array(m.get_signal_map('shear'))
    psf = np.array(m.get_signal_map('PSF'))
    mask = m.get_mask()
    goodpix = mask > 0
    es = get_es()

    assert sh.shape == (2, hp.nside2npix(32))
    assert psf.shape == (2, hp.nside2npix(32))
    # for constant ellipticities the sh map should be 0
    # as we remove the mean ellipticity of the set
    assert np.all(np.fabs(sh) < 1E-5)
    assert np.all((np.fabs(psf[0])-es)[goodpix] < 1E-5)
    assert np.all((np.fabs(psf[1])-es)[goodpix] < 1E-5)


def test_get_mask():
    m = get_mapper()
    mask = m.get_mask()
    assert len(mask) == hp.nside2npix(32)
    # One item per pix
    goodpix = mask > 0
    mask = mask[goodpix]
    assert np.all(np.fabs(mask-1) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    mask = m.get_mask()
    goodpix = mask > 0
    fsk = len(mask[goodpix])/len(mask)
    aa = fsk*hp.nside2pixarea(32)
    sh = m.get_nl_coupled()
    assert np.all(sh[0][:2] == 0)
    assert np.fabs(np.mean(sh[0][2:])) < 1E-5

    psf = m.get_nl_coupled('PSF')
    psfp = np.mean(np.arange(4)**2)*aa
    assert np.all(psf[0][:2] == 0)
    assert np.fabs(np.mean(psf[0][2:])-psfp) < 1E-5


def test_get_dtype():
    m = get_mapper()
    dtype = m.get_dtype()
    assert dtype == 'galaxy_shear'


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    assert np.all(z == 0.6 * np.ones(m.npix))
    assert np.all(nz == (m.zbin + 1) * np.ones(m.npix))
