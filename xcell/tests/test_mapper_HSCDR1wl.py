import numpy as np
import xcell as xc
import healpy as hp
import os
import glob
from astropy.table import Table, hstack


def make_hsc_data():
    d = Table.read('xcell/tests/data/catalog.fits')
    ng = len(d)
    # Add new columns
    falsarr = np.zeros(ng, dtype=bool)
    truearr = np.ones(ng, dtype=bool)
    zerarr = np.zeros(ng)
    onearr = np.ones(ng)
    isn = 'ishape_hsm_regauss'
    d[f'{isn}_flags'] = falsarr
    d[f'{isn}_resolution'] = onearr*0.5
    d[f'{isn}_sigma'] = onearr*0.1
    d[f'{isn}_e1'] = d['e1'].copy()
    d[f'{isn}_e2'] = d['e2'].copy()
    d[f'{isn}_derived_shear_bias_c1'] = zerarr
    d[f'{isn}_derived_shear_bias_c2'] = zerarr
    d[f'{isn}_derived_shear_bias_m'] = zerarr
    d[f'{isn}_derived_rms_e'] = zerarr
    d[f'{isn}_derived_shape_weight'] = onearr
    imsk = 'iflags_pixel_bright'
    d[f'{imsk}_object_center'] = falsarr
    d[f'{imsk}_object_any'] = falsarr
    d['wl_fulldepth_fullcolor'] = truearr
    d['clean_photometry'] = truearr
    d['pz_best_eab'] = onearr*0.5
    d['icmodel_mag'] = onearr*23
    d['a_i'] = zerarr
    d['iblendedness_abs_flux'] = zerarr
    for m in ['g', 'r', 'i', 'z', 'y']:
        d[f'{m}cmodel_flux'] = 11*onearr
        d[f'{m}cmodel_flux_err'] = onearr
    d['iclassification_extendedness'] = onearr
    d.write("xcell/tests/data/hsc_catalog.fits",
            overwrite=True)


def clean_hsc_data():
    os.remove("xcell/tests/data/hsc_catalog.fits")


def get_config():
    return {'data_catalogs': [['xcell/tests/data/hsc_catalog.fits']],
            'path_lite': 'xcell/tests/data/',
            'depth_cut': 24.5,
            'z_edges': [-1, 100],
            'shear_mod_thr': 10,
            'bin_name': 'bin_test',
            'fname_cosmos': 'nothing',
            'fname_cosmos_ph': 'nothing',
            'nside': 32, 'mask_name': 'mask'}


def remove_lite(plite):
    flite = glob.glob(plite + 'HSCDR1wl*')
    for f in flite:
        os.remove(f)


def test_smoke():
    make_hsc_data()
    c = get_config()
    m = xc.mappers.MapperHSCDR1wl(c)
    m.get_catalog()
    clean_hsc_data()
    remove_lite(c['path_lite'])
    assert False


def get_mapper():
    return xc.mappers.MapperDESY1wl(get_config())


def atest_smoke():
    # Checks for errors in summoning mapper
    get_mapper()


def get_es():
    # calculates an estimate for the signal of a periodic
    # set of ellipticities in the sky
    # (see test_get_signal_map())
    npix = hp.nside2npix(32)
    return np.repeat(np.array([np.arange(4)]), npix//4,
                     axis=0).flatten()


def atest_load_catalog():
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


def atest_lite():
    config = get_config()
    plite = 'xcell/tests/data/'
    config['path_lite'] = plite
    remove_lite(plite)
    # Initialize mapper
    m = xc.mappers.MapperDESY1wl(config)
    # Get maps and catalog
    cat = m.get_catalog()
    s = m.get_signal_map()
    psf = m.get_signal_map('PSF')
    mask = m.get_mask()
    nl_cp = m.get_nl_coupled()
    nl_cp_psf = m.get_nl_coupled('PSF')

    # Check lite files have been created
    zbin = config['zbin']
    nside = config['nside']

    for fname in [f'catalog_lite_zbin{zbin}.fits',
                  f'signal_map_shear_e1_zbin{zbin}_ns{nside}.fits.gz',
                  f'signal_map_shear_e2_zbin{zbin}_ns{nside}.fits.gz',
                  f'signal_map_PSF_e1_zbin{zbin}_ns{nside}.fits.gz',
                  f'signal_map_PSF_e2_zbin{zbin}_ns{nside}.fits.gz',
                  f'signal_map_PSF_e2_zbin{zbin}_ns{nside}.fits.gz',
                  f'mask_zbin{zbin}_ns{nside}.fits.gz',
                  f'shear_w2s2_zbin{zbin}_ns{nside}.fits.gz',
                  f'PSF_w2s2_zbin{zbin}_ns{nside}.fits.gz']:
        assert os.path.isfile(os.path.join(plite, "DESwlMETACAL_" + fname))

    # Check we recover the same mas and catalog
    # Non-exsisting fits files - read from lite
    config['data_catalogs'] = 'whatever'
    m_lite = xc.mappers.MapperDESY1wl(config)
    cat_from_lite = m_lite.get_catalog()
    s_from_lite = m_lite.get_signal_map()
    psf_from_lite = m_lite.get_signal_map('PSF')
    mask_from_lite = m_lite.get_mask()
    nl_cp_from_lite = m_lite.get_nl_coupled()
    nl_cp_psf_from_lite = m_lite.get_nl_coupled('PSF')
    assert np.all(cat == cat_from_lite)
    assert np.all(np.array(s) == np.array(s_from_lite))
    assert np.all(np.array(psf) == np.array(psf_from_lite))
    assert np.all(mask == mask_from_lite)
    assert np.all(nl_cp == nl_cp_from_lite)
    assert np.all(nl_cp_psf == nl_cp_psf_from_lite)

    # Clean lite
    remove_lite(plite)


def atest_get_signal_map():
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


def atest_get_mask():
    m = get_mapper()
    mask = m.get_mask()
    assert len(mask) == hp.nside2npix(32)
    # One item per pix
    goodpix = mask > 0
    mask = mask[goodpix]
    assert np.all(np.fabs(mask-1) < 1E-5)


def atest_get_nl_coupled():
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


def atest_get_dtype():
    m = get_mapper()
    dtype = m.get_dtype()
    assert dtype == 'galaxy_shear'


def atest_get_nz():
    m = get_mapper()
    z, nz = m.get_nz()
    assert np.all(z == 0.6 * np.ones(m.npix))
    assert np.all(nz == (m.zbin + 1) * np.ones(m.npix))
