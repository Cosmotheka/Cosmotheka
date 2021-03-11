import numpy as np
import xcell as xc
import healpy as hp
import os


def get_config():
    return {'data_catalogs': ['xcell/tests/data/catalog.fits',
                              'xcell/tests/data/catalog_stars.fits'],
            'file_nz': 'xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
            'zbin': 0, 'nside': 32, 'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperKV450(get_config())


def test_smoke():
    get_mapper()


def get_es():
    npix = hp.nside2npix(32)
    return np.repeat(np.array([np.arange(4)]), npix//4,
                     axis=0).flatten()


def test_lite():
    config = get_config()
    config['path_lite'] = 'xcell/tests/data/'
    ifile = 0
    while os.path.isfile(f'xcell/tests/data/KV450_lite_cat_zbin{ifile}.fits'):
        os.remove(f'xcell/tests/data/KV450_lite_cat_zbin{ifile}.fits')
        ifile += 1
    m = xc.mappers.MapperKV450(config)
    m.get_catalog()
    assert os.path.isfile('xcell/tests/data/KV450_lite_cat_zbin0.fits')

    # Non-exsisting fits files - read from lite
    config['data_catalogs'] = ['whatever', 'whatever']
    xc.mappers.MapperKV450(config)


def test_get_signal_map():
    m = get_mapper()
    sh = np.array(m.get_signal_map('shear'))
    psf = np.array(m.get_signal_map('PSF'))
    star = np.array(m.get_signal_map('stars'))
    es = get_es()
    assert sh.shape == (2, hp.nside2npix(32))
    assert psf.shape == (2, hp.nside2npix(32))
    assert star.shape == (2, hp.nside2npix(32))
    assert np.all(np.fabs(-sh+(np.mean(es)-es)/(1+m.m[0])) < 1E-5)
    assert np.all(np.fabs(-psf-es) < 1E-5)
    assert np.all(np.fabs(-star-es) < 1E-5)


def test_get_mask():
    m = get_mapper()
    sh = m.get_mask('shear')
    psf = m.get_mask('PSF')
    star = m.get_mask('stars')
    assert len(sh) == len(psf) == len(star) == hp.nside2npix(32)
    assert np.all(np.fabs(sh-2) < 1E-5)
    assert np.all(np.fabs(psf-2) < 1E-5)
    assert np.all(np.fabs(star-2) < 1E-5)


def test_get_nl_coupled():
    m = get_mapper()
    aa = hp.nside2pixarea(32)

    sh = m.get_nl_coupled()
    shp = 4*np.std(np.arange(4))**2*aa/(1+m.m[0])**2
    assert np.all(sh[0][:2] == 0)
    assert np.fabs(np.mean(sh[0][2:])-shp) < 1E-5

    psf = m.get_nl_coupled('PSF')
    psfp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(psf[0][:2] == 0)
    assert np.fabs(np.mean(psf[0][2:])-psfp) < 1E-5

    star = m.get_nl_coupled('stars')
    starp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(star[0][:2] == 0)
    assert np.fabs(np.mean(star[0][2:])-starp) < 1E-5
