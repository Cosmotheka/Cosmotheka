import numpy as np
import xcell as xc
import healpy as hp
import os
import glob
from astropy.table import Table


def make_hsc_nz_data():
    ng = 1000
    onearr = np.ones(ng)
    ids = np.arange(ng, dtype=int)
    zs = 0.3 + 0.02*np.random.randn(ng)
    c_s = Table()
    c_s['S17a_objid'] = ids
    c_s['SOM_weight'] = onearr
    c_s['weight_source'] = onearr
    c_s['COSMOS_photoz'] = zs
    c_s.write("xcell/tests/data/hsc_cosmos_catalog.fits",
              overwrite=True)
    c_p = Table()
    c_p['ID'] = ids
    c_p['PHOTOZ_BEST'] = onearr*0.5
    c_p.write("xcell/tests/data/hsc_photoz_catalog.fits",
              overwrite=True)


def clean_hsc_nz_data():
    for fn in ["xcell/tests/data/hsc_cosmos_catalog.fits",
               "xcell/tests/data/hsc_photoz_catalog.fits"]:
        if os.path.isfile(fn):
            os.remove(fn)


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
    # Factor 2 is there because these are uncorrected
    # ellipticities (they get corrected by 1/(2*response))
    d[f'{isn}_e1'] = d['e1'].copy()*2
    d[f'{isn}_e2'] = d['e2'].copy()*2
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
    fn = "xcell/tests/data/hsc_catalog.fits"
    if os.path.isfile(fn):
        os.remove(fn)


def get_config():
    return {'data_catalogs': [['xcell/tests/data/hsc_catalog.fits']],
            'path_rerun': 'xcell/tests/data/',
            'depth_cut': 24.5,
            'z_edges': [0., 0.5],
            'shear_mod_thr': 10,
            'bin_name': 'bin_test',
            'fname_cosmos': 'xcell/tests/data/hsc_cosmos_catalog.fits',
            'fnames_cosmos_ph': ['xcell/tests/data/hsc_photoz_catalog.fits'],
            'nside': 32, 'mask_name': 'mask', 'coords': 'C'}


def remove_rerun(prerun):
    frerun = glob.glob(prerun + 'HSCDR1wl*')
    for f in frerun:
        os.remove(f)
    fn = prerun + 'mask_mask_coordC_ns32.fits.gz'
    if os.path.isfile(fn):
        os.remove(fn)


def test_smoke():
    make_hsc_data()
    c = get_config()
    m = xc.mappers.MapperHSCDR1wl(c)
    m.get_catalog()
    m.get_signal_map()
    m.get_nl_coupled()
    clean_hsc_data()
    remove_rerun(c['path_rerun'])


def test_rerun():
    make_hsc_data()
    make_hsc_nz_data()
    c = get_config()
    prerun = c['path_rerun']
    remove_rerun(prerun)

    # Initialize mapper
    m = xc.mappers.MapperHSCDR1wl(c)
    # Get catalog, maps and n(z)
    cat = m.get_catalog()
    mask = m.get_mask()
    s = m.get_signal_map()
    nl_cp = m.get_nl_coupled()
    zz, nzz = m.get_nz()

    # Check rerun files have been created
    bn = c['bin_name']
    nside = c['nside']

    for fname in [f'{bn}.fits',
                  f'signal_{bn}_coordC_ns{nside}.fits.gz',
                  f'w2s2_{bn}_coordC_ns{nside}.fits.gz',
                  f'nz_{bn}.npz']:
        fname_full = os.path.join(prerun, "HSCDR1wl_" + fname)
        assert os.path.isfile(fname_full)
    fname_full = os.path.join(prerun, "mask_mask_coordC_ns32.fits.gz")
    assert os.path.isfile(fname_full)

    # Check we recover the same mask and catalog
    # Non-exsisting fits files - read from rerun
    c['data_catalogs'] = [['whatever']]
    m_rerun = xc.mappers.MapperHSCDR1wl(c)
    cat_from_rerun = m_rerun.get_catalog()
    s_from_rerun = m_rerun.get_signal_map()
    mask_from_rerun = m_rerun.get_mask()
    nl_cp_from_rerun = m_rerun.get_nl_coupled()
    zz_from_rerun, nzz_from_rerun = m_rerun.get_nz()
    assert np.all(cat == cat_from_rerun)
    assert np.all(np.array(s) == np.array(s_from_rerun))
    assert np.all(mask == mask_from_rerun)
    assert np.all(nl_cp == nl_cp_from_rerun)
    assert np.all(zz == zz_from_rerun)
    assert np.all(nzz == nzz_from_rerun)

    # Clean rerun
    clean_hsc_data()
    clean_hsc_nz_data()
    remove_rerun(prerun)


def test_get_signal_map():
    make_hsc_data()
    c = get_config()
    prerun = c['path_rerun']
    m = xc.mappers.MapperHSCDR1wl(c)
    sh = np.array(m.get_signal_map())
    mask = m.get_mask()
    assert sh.shape == (2, hp.nside2npix(32))
    assert np.all(mask == 1.0)
    assert np.all((sh[0]-1.0) < 1E-5)
    assert np.all((sh[1]+1.0) < 1E-5)
    clean_hsc_data()
    remove_rerun(prerun)


def test_get_nl_coupled():
    make_hsc_data()
    c = get_config()
    prerun = c['path_rerun']
    m = xc.mappers.MapperHSCDR1wl(c)
    nlc = m.get_nl_coupled()
    apix = hp.nside2pixarea(32)
    assert np.all(nlc[0][:2] == 0)
    assert np.all(np.fabs(nlc[0, 2:]-apix) < 1E-5)
    assert np.all(np.fabs(nlc[1, 2:]) < 1E-5)
    assert np.all(np.fabs(nlc[2, 2:]) < 1E-5)
    assert np.all(np.fabs(nlc[3, 2:]-apix) < 1E-5)
    clean_hsc_data()
    remove_rerun(prerun)


def test_get_dtype_and_spin():
    c = get_config()
    m = xc.mappers.MapperHSCDR1wl(c)
    assert m.get_dtype() == 'galaxy_shear'
    assert m.get_spin() == 2


def test_get_nz():
    make_hsc_nz_data()
    c = get_config()
    m = xc.mappers.MapperHSCDR1wl(c)
    z, nz = m.get_nz()
    cat = Table.read(c['fname_cosmos'])
    hz, bz = np.histogram(cat['COSMOS_photoz'],
                          bins=100, range=[0., 4.],
                          density=True)
    dndz = hz * len(cat)
    zm = 0.5*(bz[1:] + bz[:-1])
    assert np.all(dndz == nz)
    assert np.all(zm == z)
    clean_hsc_nz_data()
    remove_rerun(c['path_rerun'])
