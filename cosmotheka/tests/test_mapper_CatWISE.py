import numpy as np
import cosmotheka as xc
import healpy as hp
import os
import pytest
from astropy.table import Table


def get_config(corr=True):
    data_path = 'cosmotheka/tests/data/'
    return {'data_catalog': data_path+'catalog_CatWISE.fits',
            'mask_sources': data_path+'MASKS_exclude_master_final.fits',
            'nside': 32, 'mask_name': 'mask', 'coords': 'C',
            'apply_ecliptic_correction': corr}


def make_fake_data():
    data_path = 'cosmotheka/tests/data/'
    nside = 32
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    flux = 9 + (16.4 - 9)*np.random.rand(npix)
    c = Table({'ra': ra,
               'dec': dec,
               'w1': flux})
    c.write(data_path+'catalog_CatWISE.fits', overwrite=True)


def clean_fake_data():
    for fn in ['catalog_CatWISE.fits',
               'mask_mask_coordC_ns32.fits.gz']:
        fname = f'cosmotheka/tests/data/{fn}'
        if os.path.isfile(fname):
            os.remove(fname)


def test_basic():
    make_fake_data()
    config = get_config()
    config['GLAT_max_deg'] = 0.
    m = xc.mappers.MapperCatWISE(config)
    c = m.get_catalog()
    assert len(c) == hp.nside2npix(32)
    clean_fake_data()


def test_get_mask():
    config = get_config()
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperCatWISE(config)
    d = m.get_mask()
    assert np.all(np.fabs(d-1) == 0)


@pytest.mark.parametrize('corr', [True, False])
def test_get_nmap_data(corr):
    make_fake_data()
    config = get_config(corr)
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperCatWISE(config)
    nmap = m._get_nmap_data()

    assert nmap.size == hp.nside2npix(m.nside)
    if corr is False:
        assert np.all(nmap == 1)
    else:
        # It's 0 because the ecliptic_correction is -11.28, and nmap > 0.
        nmap_ref = 1 + m._get_ecliptic_correction()
        assert np.all(nmap == nmap_ref)
    clean_fake_data()


@pytest.mark.parametrize('corr', [True, False])
def test_get_signal_map(corr):
    make_fake_data()
    config = get_config(corr)
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperCatWISE(config)
    d = m.get_signal_map()
    d = np.array(d)
    assert d.shape == (1, hp.nside2npix(m.nside))
    if corr is False:
        assert np.all(np.fabs(d) < 1E-5)
    else:
        nmap_ref = 1 + m._get_ecliptic_correction()
        dref = nmap_ref / np.mean(nmap_ref) - 1
        assert np.all(np.fabs(d/dref - 1) < 1E-5)
    clean_fake_data()


@pytest.mark.parametrize('corr', [True, False])
def test_get_nl_coupled(corr):
    make_fake_data()
    config = get_config(corr)
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperCatWISE(config)
    nl = m.get_nl_coupled()
    nl = np.array(nl)
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    assert nl.shape == (1, 3*m.nside)
    if corr is False:
        nl_pred = pix_area
        assert np.all(np.fabs(nl/nl_pred-1) < 1E-10)
    else:
        nmap_ref = 1 + m._get_ecliptic_correction()

        nmean_srad = np.mean(nmap_ref) / pix_area
        nmean_srad_uncorr = 1 / pix_area

        nl_pred = nmean_srad_uncorr / nmean_srad**2
        assert np.all(np.fabs(nl/nl_pred-1) < 1E-10)
    clean_fake_data()


def test_ecliptic_correction():
    config = get_config()
    m = xc.mappers.MapperCatWISE(config)
    d = m._get_ecliptic_correction()
    d = np.array(d)
    pixarea_deg2 = (hp.nside2resol(m.nside, arcmin=True)/60)**2
    assert d.shape[0] == hp.nside2npix(m.nside)
    assert np.all(np.fabs(d) < 0.0513 * np.abs(90.) * pixarea_deg2)


def test_get_dtype():
    config = get_config()
    m = xc.mappers.MapperCatWISE(config)
    assert m.get_dtype() == 'galaxy_density'


def test_get_spin():
    config = get_config()
    m = xc.mappers.MapperCatWISE(config)
    assert m.get_spin() == 0


def test_rerun():
    config = get_config()
    config['path_rerun'] = 'cosmotheka/tests/data/'
    config.pop('mask_sources')
    m = xc.mappers.MapperCatWISE(config)
    msk = m.get_mask()
    fn = 'cosmotheka/tests/data/mask_mask_coordC_ns32.fits.gz'
    mskb = hp.read_map(fn)
    assert (msk == mskb).all()
    os.remove(fn)
    clean_fake_data()
