import numpy as np
import cosmotheka as xc
import healpy as hp
import os
import glob
import h5py
import shutil
import pytest


OUTDIR = 'cosmotheka/tests/data/tmp_decade/'
CAT_PATH = OUTDIR + 'decade_catalog.hdf5'
NZ_PATH = OUTDIR + 'NGC_n_of_z.npy'
ZGRID_PATH = OUTDIR + 'z_grid.npy'
NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
R1, R2 = 0.8, 0.6
PSF1, PSF2 = 0.01, 0.02


def get_data():
    # One galaxy per pixel
    ipix = np.arange(NPIX)
    ra, dec = hp.pix2ang(NSIDE, ipix, lonlat=True)
    e1 = (ipix % 4).astype(float)
    e2 = 0.5*(ipix % 3).astype(float)
    # Objects spread over the 4 bins, some unselected
    sel = (ipix // 4) % 4 + 1
    sel[ipix % 7 == 0] = 0
    # Remove some objects from the 1P selection to get a non-zero Rs
    sel_1p = sel.copy()
    sel_1p[e1 == 3] = 0
    w = 1 + (ipix % 2).astype(float)
    return {'RA': ra, 'DEC': dec,
            'MCAL_G_1_NOSHEAR': e1, 'MCAL_G_2_NOSHEAR': e2,
            'MCAL_G_1_1P': e1 + 0.01*R1, 'MCAL_G_1_1M': e1 - 0.01*R1,
            'MCAL_G_2_2P': e2 + 0.01*R2, 'MCAL_G_2_2M': e2 - 0.01*R2,
            'MCAL_W_NOSHEAR': w, 'MCAL_W_1P': w, 'MCAL_W_1M': w,
            'MCAL_W_2P': w, 'MCAL_W_2M': w,
            'MCAL_SEL_NOSHEAR': sel, 'MCAL_SEL_1P': sel_1p,
            'MCAL_SEL_1M': sel, 'MCAL_SEL_2P': sel, 'MCAL_SEL_2M': sel,
            'MCAL_PSF_G_1_NOSHEAR': PSF1*np.ones(NPIX),
            'MCAL_PSF_G_2_NOSHEAR': PSF2*np.ones(NPIX)}


def get_expected(region='NGC', zbin=0):
    # Independent calculation of the calibrated ellipticities
    d = get_data()
    ra_ngp, dec_ngp = np.radians([193., 27.])
    ra, dec = np.radians(d['RA']), np.radians(d['DEC'])
    b = np.sin(dec)*np.sin(dec_ngp) + \
        np.cos(dec)*np.cos(dec_ngp)*np.cos(ra-ra_ngp)
    reg = b > 0 if region == 'NGC' else b <= 0
    good = reg & (d['MCAL_SEL_NOSHEAR'] == zbin+1)
    good_1p = reg & (d['MCAL_SEL_1P'] == zbin+1)
    w = d['MCAL_W_NOSHEAR']
    e1 = d['MCAL_G_1_NOSHEAR']
    e2 = d['MCAL_G_2_NOSHEAR']
    Rs11 = (np.average(e1[good_1p], weights=w[good_1p]) -
            np.average(e1[good], weights=w[good]))/0.02
    R = 0.5*(R1 + Rs11 + R2)
    e1c = (e1[good] - np.average(e1[good], weights=w[good]))/R
    e2c = (e2[good] - np.average(e2[good], weights=w[good]))/R
    return good, e1c, e2c, w[good]


def gen_catalog():
    with h5py.File(CAT_PATH, mode='w') as f:
        for k, v in get_data().items():
            f.create_dataset(k, data=v)


def gen_nz():
    z = np.linspace(0, 2, 101)
    zgrid = np.array([z[:-1], 0.5*(z[1:]+z[:-1]), z[1:]]).T
    nz = np.array([(i+1)*np.ones(100) for i in range(4)])
    np.save(ZGRID_PATH, zgrid)
    np.save(NZ_PATH, nz)


def remove_rerun(prerun):
    frerun = glob.glob(prerun + 'DECADEwl*.fits*')
    for f in frerun:
        os.remove(f)
    fn = prerun + 'mask_mask_coordC_ns32.fits.gz'
    if os.path.isfile(fn):
        os.remove(fn)


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)
    gen_catalog()
    gen_nz()


def teardown_module():
    shutil.rmtree(OUTDIR, ignore_errors=True)


@pytest.fixture
def config():
    return {'data_catalog': CAT_PATH,
            'file_nz': NZ_PATH,
            'file_zgrid': ZGRID_PATH,
            'region': 'NGC',
            'mode': 'shear',
            'zbin': 0, 'nside': NSIDE, 'mask_name': 'mask',
            'coords': 'C'}


@pytest.fixture
def mapper(config):
    return xc.mappers.MapperDECADEwl(config)


def test_smoke(config):
    m = xc.mappers.MapperDECADEwl(config)
    assert m.map_name == 'DECADEwl_NGC_bin0'


@pytest.mark.parametrize("key,value", [('region', 'all'),
                                       ('zbin', 4),
                                       ('zbin', -1)])
def test_init_errors(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        xc.mappers.MapperDECADEwl(config)


@pytest.mark.parametrize("mode", ["PSF", "shear", "fail"])
def test_set_mode(mapper, mode):
    if mode == "PSF":
        assert mapper._set_mode(mode) == ('PSF_e1', 'PSF_e2', 'PSF')
    elif mode == "shear":
        assert mapper._set_mode(mode) == ('e1', 'e2', 'shear')
    else:
        with pytest.raises(ValueError):
            mapper._set_mode(mode)


@pytest.mark.parametrize("region,zbin", [('NGC', 0), ('NGC', 1),
                                         ('SGC', 0), ('SGC', 1)])
def test_get_catalog(config, region, zbin):
    config['region'] = region
    config['zbin'] = zbin
    m = xc.mappers.MapperDECADEwl(config)
    cat = m.get_catalog()
    good, e1, e2, w = get_expected(region, zbin)
    d = get_data()
    assert len(cat) == np.sum(good)
    assert np.all(cat['RA'] == d['RA'][good])
    assert np.all(cat['DEC'] == d['DEC'][good])
    assert np.allclose(cat['e1'], e1)
    assert np.allclose(cat['e2'], e2)
    assert np.all(cat['PSF_e1'] == PSF1)
    assert np.all(cat['PSF_e2'] == PSF2)
    assert np.all(cat['weight'] == w)
    # Calibrated ellipticities have zero weighted mean
    assert np.fabs(np.average(cat['e1'], weights=cat['weight'])) < 1E-10
    assert np.fabs(np.average(cat['e2'], weights=cat['weight'])) < 1E-10


def test_get_mask(mapper):
    mask = mapper.get_mask()
    good, _, _, w = get_expected()
    assert len(mask) == NPIX
    assert np.all(mask[good] == w)
    assert np.all(mask[~good] == 0)


@pytest.mark.parametrize("mode", ["PSF", "shear"])
def test_get_signal_map(mapper, mode):
    good, e1, e2, w = get_expected()
    mp = mapper.get_signal_map(mode)
    assert mp.shape == (2, NPIX)
    if mode == 'shear':
        assert np.allclose(mp[0][good], -e1)
        assert np.allclose(mp[1][good], e2)
    else:
        assert np.allclose(mp[0][good], -PSF1)
        assert np.allclose(mp[1][good], PSF2)
    assert np.all(mp[:, ~good] == 0)
    # Check pre-loaded map
    assert np.all(mapper.get_signal_map(mode) == mp)


@pytest.mark.parametrize("mode", ["PSF", "shear"])
def test_get_nl_coupled(mapper, mode):
    good, e1, e2, w = get_expected()
    if mode == 'PSF':
        e1 = PSF1
        e2 = PSF2
    nlp = hp.nside2pixarea(NSIDE)*np.sum(w**2*0.5*(e1**2+e2**2))/NPIX
    nl = mapper.get_nl_coupled(mode)
    assert nl.shape == (4, 3*NSIDE)
    assert np.all(nl[0][:2] == 0)
    assert np.allclose(nl[0][2:], nlp)
    assert np.allclose(nl[3], nl[0])
    assert np.all(nl[1] == 0)
    assert np.all(nl[2] == 0)


def test_get_nz(config):
    config['zbin'] = 2
    m = xc.mappers.MapperDECADEwl(config)
    z, nz = m.get_nz()
    zgrid = np.load(ZGRID_PATH)
    assert np.allclose(z, zgrid[:, 1])
    assert np.all(nz == 3)
    z2, _ = m.get_nz(dz=0.1)
    assert np.allclose(z2, z+0.1)


def test_get_nmt_catalog_field(mapper):
    f = mapper._get_nmt_catalog_field()
    assert isinstance(f, xc.mappers.mapper_DECADEwl.nmt.NmtFieldCatalog)


def test_get_dtype(mapper):
    assert mapper.get_dtype() == 'galaxy_shear'


def test_get_spin(mapper):
    assert mapper.get_spin() == 2


def test_rerun(config):
    config['path_rerun'] = OUTDIR
    remove_rerun(OUTDIR)
    m = xc.mappers.MapperDECADEwl(config)
    s = m.get_signal_map()
    psf = m.get_signal_map('PSF')
    mask = m.get_mask()
    nl = m.get_nl_coupled()
    nl_psf = m.get_nl_coupled('PSF')

    # All bins of the catalog are saved
    for ib in range(4):
        assert os.path.isfile(os.path.join(OUTDIR,
                                           f'DECADEwl_NGC_bin{ib}_cat.fits'))
    for fname in [f'signal_map_shear_coordC_ns{NSIDE}.fits.gz',
                  f'signal_map_PSF_coordC_ns{NSIDE}.fits.gz',
                  f'shear_w2s2_coordC_ns{NSIDE}.fits.gz',
                  f'PSF_w2s2_coordC_ns{NSIDE}.fits.gz']:
        assert os.path.isfile(os.path.join(OUTDIR,
                                           f"DECADEwl_NGC_bin0_{fname}"))
    assert os.path.isfile(os.path.join(OUTDIR,
                                       f'mask_mask_coordC_ns{NSIDE}.fits.gz'))

    # Non-existing catalog - read from rerun
    config['data_catalog'] = 'whatever'
    m2 = xc.mappers.MapperDECADEwl(config)
    assert np.allclose(m.get_catalog()['e1'], m2.get_catalog()['e1'])
    assert np.all(np.array(s) == np.array(m2.get_signal_map()))
    assert np.all(np.array(psf) == np.array(m2.get_signal_map('PSF')))
    assert np.all(mask == m2.get_mask())
    # Allow for round-off differences when averaging maps read from file
    assert np.allclose(nl, m2.get_nl_coupled(), rtol=1E-10, atol=0)
    assert np.allclose(nl_psf, m2.get_nl_coupled('PSF'), rtol=1E-10, atol=0)

    # Other bins are read from rerun too
    config['zbin'] = 1
    m3 = xc.mappers.MapperDECADEwl(config)
    good, e1, _, _ = get_expected('NGC', 1)
    assert np.allclose(m3.get_catalog()['e1'], e1)

    remove_rerun(OUTDIR)
