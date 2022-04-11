import numpy as np
import xcell as xc
import healpy as hp
import os
from astropy.table import Table


def get_config():
    return {'data_catalog': 'xcell/tests/data/catalog_nvss.fits',
            'mask_sources': 'xcell/tests/data/source_masks_nvss.txt',
            'nside': 32, 'mask_name': 'mask', 'coords': 'C',
            'redshift_catalog': 'xcell/tests/data/redshift_catalog_nvss.fits'}


def make_fake_data():
    nside = 32
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    flux = 10+(1000-10)*np.random.rand(npix)
    c = Table({'RAJ2000': ra,
               'DEJ2000': dec,
               'S1_4': flux})
    c.write('xcell/tests/data/catalog_nvss.fits', overwrite=True)

    sources = 1500
    max_redshift = 5
    redshift = max_redshift*np.random.rand(sources)
    # The flux condition 10mJy<Flux<1000mJy can be translated
    # into -2<'itot_1400'<0 since mJy = 10***(3+'itot_1400')
    flux_redshift = -2+2*np.random.rand(sources)
    b = Table({'redshift': redshift,
              'itot_1400': flux_redshift})
    b.write('xcell/tests/data/redshift_catalog_nvss.fits', overwrite=True)


def clean_fake_data():
    os.remove('xcell/tests/data/catalog_nvss.fits')
    os.remove('xcell/tests/data/redshift_catalog_nvss.fits')


def test_basic():
    make_fake_data()
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    c = m.get_catalog()
    # The sentence below is testing that sky fraction covered by NVSS
    # is fsky ≃ 0.75  following the angular and flux conditions
    # on ArXiv 1901.08357
    assert (0.75 < len(c)/hp.nside2npix(32) < 0.76)
    clean_fake_data()


def test_get_mask():
    # Nothing should get masked
    config = get_config()
    config['DEC_min_deg'] = -90.
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperNVSS(config)
    d = m.get_mask()
    assert np.all(np.fabs(d-1) == 0)

    # Again, nothing should get masked
    config = get_config()
    config['mask_file'] = 'xcell/tests/data/map.fits'
    m = xc.mappers.MapperNVSS(config)
    d = m.get_mask()
    assert np.all(np.fabs(d-1) == 0)

    # Now sources should get masked
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    d = m.get_mask()
    ra, dec, _ = np.loadtxt('xcell/tests/data/source_masks_nvss.txt',
                            unpack=True)
    ipix = hp.ang2pix(32, ra, dec, lonlat=True)
    assert np.all(d[ipix] == 0)


def test_get_signal_map():
    make_fake_data()
    config = get_config()
    config['DEC_min_deg'] = -90.
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperNVSS(config)
    d = m.get_signal_map()
    d = np.array(d)
    assert d.shape == (1, hp.nside2npix(m.nside))
    assert np.all(np.fabs(d) < 1E-15)
    clean_fake_data()


def test_get_nz():
    sources = 1500
    make_fake_data()
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    z, nz = m.get_nz()
    total = np.sum(nz)
    assert total == sources
    clean_fake_data()


def test_get_nl_coupled():
    make_fake_data()
    config = get_config()
    config['DEC_min_deg'] = -90.
    config['GLAT_max_deg'] = 0.
    config.pop('mask_sources')
    m = xc.mappers.MapperNVSS(config)
    nl = m.get_nl_coupled()
    nl = np.array(nl)
    pix_area = 4*np.pi/hp.nside2npix(m.nside)
    nl_pred = hp.nside2npix(32)
    nl_pred *= pix_area**2/(4*np.pi)
    assert nl.shape == (1, 3*m.nside)
    assert np.all(np.fabs(nl/nl_pred-1) < 1E-10)
    clean_fake_data()


def test_get_dtype():
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    assert m.get_dtype() == 'galaxy_density'


def test_get_spin():
    config = get_config()
    m = xc.mappers.MapperNVSS(config)
    assert m.get_spin() == 0
