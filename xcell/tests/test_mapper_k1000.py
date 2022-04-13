import numpy as np
import xcell as xc
import healpy as hp
import os
import glob


def get_config(mode='shear', w_stars=False):
    if w_stars:
        fname = 'xcell/tests/data/catalog_stars.fits'
    else:
        fname = 'xcell/tests/data/catalog.fits'
    return {'data_catalog': fname,
            'file_nz': 'xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
            'zbin': 0, 'nside': 32, 'mask_name': 'mask',
            'coords': 'C', 'mode': mode,
            'e1_flag': 'bias_corrected_e1',
            'e2_flag': 'bias_corrected_e2'}


def get_mapper(mode='shear', w_stars=False):
    return xc.mappers.MapperKiDS1000(get_config(mode=mode,
                                                w_stars=w_stars))


def test_smoke():
    get_mapper()


def get_es():
    npix = hp.nside2npix(32)
    return np.repeat(np.array([np.arange(4)]), npix//4,
                     axis=0).flatten()


def remove_rerun(predir):
    frerun = glob.glob(predir + 'KiDS1000_*.fits*')
    for f in frerun:
        os.remove(f)

    fn = predir + 'mask_mask_coordC_ns32.fits.gz'
    if os.path.isfile(fn):
        os.remove(fn)


def test_rerun():
    predir = 'xcell/tests/data/'
    config = get_config()
    config['path_rerun'] = predir
    prefix = f'{predir}KiDS1000'
    remove_rerun(predir)
    m = xc.mappers.MapperKiDS1000(config)
    m.get_catalog()
    assert os.path.isfile(f'{prefix}_cat_bin0.fits')
    map1 = np.array(m.get_signal_map())
    mask1 = m.get_mask()
    assert os.path.isfile(f'{prefix}_signal_shear_bin0_coordC_ns32.fits.gz')
    assert os.path.isfile(f'{predir}mask_mask_coordC_ns32.fits.gz')
    nl1 = m.get_nl_coupled()
    assert os.path.isfile(f'{prefix}_w2s2_galaxies_bin0_coordC_ns32.fits.gz')

    # Non-exsisting fits files - read from rerun
    config['data_catalog'] = 'whatever'
    m = xc.mappers.MapperKiDS1000(config)
    m.get_catalog()
    mask2 = m.get_mask()
    map2 = np.array(m.get_signal_map())
    nl2 = m.get_nl_coupled()
    print(np.amax(np.fabs(mask1-mask2)))
    print(np.amax(np.fabs(map1-map2)))
    print(np.amax(np.fabs(nl1-nl2)))
    assert np.all(np.fabs(mask1 - mask2) < 1E-5)
    assert np.all(np.fabs(map1 - map2) < 1E-5)
    assert np.all(np.fabs(nl1 - nl2) < 1E-5)
    remove_rerun(predir)


def test_get_signal_map():
    m = get_mapper(mode='PSF')
    psf = np.array(m.get_signal_map())
    m = get_mapper(mode='stars', w_stars=True)
    star = np.array(m.get_signal_map())
    m = get_mapper(mode='shear')
    sh = np.array(m.get_signal_map())
    es = get_es()
    assert sh.shape == (2, hp.nside2npix(32))
    assert psf.shape == (2, hp.nside2npix(32))
    assert star.shape == (2, hp.nside2npix(32))
    assert np.all(np.fabs(-sh+(np.mean(es)-es)/(1+m.m[0])) < 1E-5)
    assert np.all(np.fabs(-psf-es) < 1E-5)
    assert np.all(np.fabs(-star-es) < 1E-5)

    # Check pre-loaded map
    sh2 = np.array(m.get_signal_map())
    assert np.all(sh2 == sh)


def test_get_mask():
    m = get_mapper(mode='shear')
    sh = m.get_mask()
    m = get_mapper(mode='PSF')
    psf = m.get_mask()
    m = get_mapper(mode='stars', w_stars=True)
    star = m.get_mask()
    assert len(sh) == len(psf) == len(star) == hp.nside2npix(32)
    assert np.all(np.fabs(sh-2) < 1E-5)
    assert np.all(np.fabs(psf-2) < 1E-5)
    assert np.all(np.fabs(star-2) < 1E-5)


def test_get_nl_coupled():
    aa = hp.nside2pixarea(32)

    m = get_mapper(mode='shear')
    sh = m.get_nl_coupled()
    shp = 4*np.std(np.arange(4))**2*aa/(1+m.m[0])**2
    assert np.all(sh[0][:2] == 0)
    assert np.fabs(np.mean(sh[0][2:])/shp-1) < 1E-5

    m = get_mapper(mode='PSF')
    psf = m.get_nl_coupled()
    psfp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(psf[0][:2] == 0)
    assert np.fabs(np.mean(psf[0][2:])/psfp-1) < 1E-5

    m = get_mapper(mode='stars', w_stars=True)
    star = m.get_nl_coupled()
    starp = 4*np.mean(np.arange(4)**2)*aa
    assert np.all(star[0][:2] == 0)
    assert np.fabs(np.mean(star[0][2:])/starp-1) < 1E-5


def test_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    zb, nzb = np.loadtxt("xcell/tests/data/Nz_DIR_z0.1t0.3.asc",
                         unpack=True)[:2]
    assert np.all(z == zb)
    assert np.all(nz == nzb)


def test_get_dtype():
    m = get_mapper()
    assert m.get_dtype() == 'galaxy_shear'


def test_get_spin():
    m = get_mapper()
    assert m.get_spin() == 2
