import numpy as np
import xcell as xc
import healpy as hp
import os


def get_config(mode='shear'):
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog_stars.fits'],
            'file_nz': 'xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
            'zbin': 0, 'nside': 32, 'mode': mode,
            'mask_name': 'mask', 'coords': 'C'}


def get_mapper(mode='shear'):
    return xc.mappers.MapperKV450(get_config(mode=mode))


def test_smoke():
    get_mapper()


def get_es():
    npix = hp.nside2npix(32)
    return np.repeat(np.array([np.arange(4)]), npix//4,
                     axis=0).flatten()


def test_rerun():
    config = get_config()
    config['path_rerun'] = 'xcell/tests/data/'
    ifile = 0
    while os.path.isfile(f'xcell/tests/data/KV450_cat_bin{ifile}.fits'):
        os.remove(f'xcell/tests/data/KV450_cat_bin{ifile}.fits')
        ifile += 1
    m = xc.mappers.MapperKV450(config)
    m.get_catalog()
    assert os.path.isfile('xcell/tests/data/KV450_cat_bin0.fits')
    ifile = 0
    # Cleanup
    while os.path.isfile(f'xcell/tests/data/KV450_cat_bin{ifile}.fits'):
        os.remove(f'xcell/tests/data/KV450_cat_bin{ifile}.fits')
        ifile += 1

    # Non-exsisting fits files - read from rerun
    config['data_catalogs'] = ['whatever', 'whatever']
    xc.mappers.MapperKV450(config)


def test_get_signal_map():
    m = get_mapper('shear')
    sh = np.array(m.get_signal_map())
    m = get_mapper('PSF')
    psf = np.array(m.get_signal_map())
    m = get_mapper('stars')
    star = np.array(m.get_signal_map())
    es = get_es()
    assert sh.shape == (2, hp.nside2npix(32))
    assert psf.shape == (2, hp.nside2npix(32))
    assert star.shape == (2, hp.nside2npix(32))
    assert np.all(np.fabs(-sh+(np.mean(es)-es)/(1+m.m[0])) < 1E-5)
    assert np.all(np.fabs(-psf-es) < 1E-5)
    assert np.all(np.fabs(-star-es) < 1E-5)


def test_get_mask():
    m = get_mapper('shear')
    sh = m.get_mask()
    m = get_mapper()
    psf = m.get_mask()
    m = get_mapper('stars')
    star = m.get_mask()
    assert len(sh) == len(psf) == len(star) == hp.nside2npix(32)
    assert np.all(np.fabs(sh-2) < 1E-5)
    assert np.all(np.fabs(psf-2) < 1E-5)
    assert np.all(np.fabs(star-2) < 1E-5)


def test_get_nl_coupled():
    aa = hp.nside2pixarea(32)

    m = get_mapper('shear')
    sh = m.get_nl_coupled()
    shp = 4*np.std(np.arange(4))**2*aa/(1+m.m[0])**2
    assert np.all(sh[0][:2] == 0)
    assert np.fabs(np.mean(sh[0][2:])-shp) < 1E-5

    m = get_mapper('PSF')
    psf = m.get_nl_coupled()
    psfp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(psf[0][:2] == 0)
    assert np.fabs(np.mean(psf[0][2:])-psfp) < 1E-5

    m = get_mapper('stars')
    star = m.get_nl_coupled()
    starp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(star[0][:2] == 0)
    assert np.fabs(np.mean(star[0][2:])-starp) < 1E-5
