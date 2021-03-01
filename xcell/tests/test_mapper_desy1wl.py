import numpy as np
import xcell as xc
import healpy as hp
import os


def get_config():
    return {'data_cat': 'xcell/tests/data/catalog.fits',
            'zbin_cat': 'xcell/tests/data/cat_zbin.fits',
            'path_lite': 'xcell/tests/data/',
            'zbin': 1, 'nside': 32, 'mask_name': 'mask'}


def get_mapper():
    return xc.mappers.MapperDESY1wl(get_config())


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


def test_lite():
    config = get_config()
    config['path_lite'] = 'xcell/tests/data/'
    ifile = 0
    while os.path.isfile(
            f'xcell/tests/data/DESwlMETACAL_catalog_lite_zbin{ifile}.fits'):
        os.remove(
                f'xcell/tests/data/DESwlMETACAL_catalog_lite_zbin{ifile}.fits')
        ifile += 1
    xc.mappers.MapperDESY1wl(config)
    zbin = config['zbin']
    assert os.path.isfile(
            f'xcell/tests/data/DESwlMETACAL_catalog_lite_zbin{zbin}.fits')

    # Non-exsisting fits files - read from lite
    config['data_catalogs'] = 'whatever'
    xc.mappers.MapperDESY1wl(config)


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
