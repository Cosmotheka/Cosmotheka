import numpy as np
import cosmotheka as xc
import healpy as hp
import os
import glob
import shutil
import pytest
from astropy.io import fits


OUTDIR = 'cosmotheka/tests/data/tmp_kidslegacy/'
CAT_PATH = OUTDIR + 'kids_legacy_catalog.fits'
NZ_PATH = OUTDIR + 'kids_legacy_nz.fits'
NSIDE = 32
NPIX = hp.nside2npix(NSIDE)
PSF1, PSF2 = 0.01, 0.02
ZB_MID = np.array([0.2, 0.5, 0.65, 0.8, 1.0, 1.5])


def get_data():
    # One galaxy per pixel
    ipix = np.arange(NPIX)
    ra, dec = hp.pix2ang(NSIDE, ipix, lonlat=True)
    e1 = (ipix % 4).astype(float)
    e2 = 0.5*(ipix % 3).astype(float)
    # Objects spread over the 6 bins, some outside all of them
    zb = ZB_MID[(ipix // 4) % 6]
    zb[ipix % 7 == 0] = 2.5
    # Different mean ellipticities in each patch
    patch = np.where(dec > 0, 'N', 'S')
    e1[dec > 0] += 0.3
    w = 1 + (ipix % 2).astype(float)
    return {'RAJ2000': ra, 'DECJ2000': dec, 'Z_B': zb, 'PATCH': patch,
            'e1': e1, 'e2': e2,
            'PSF_e1': PSF1*np.ones(NPIX), 'PSF_e2': PSF2*np.ones(NPIX),
            'weight': w}


def get_expected(zbin=0):
    # Independent calculation of the calibrated ellipticities
    d = get_data()
    good = d['Z_B'] == ZB_MID[zbin]
    w = d['weight']
    e1 = d['e1'].copy()
    e2 = d['e2'].copy()
    for p in ['N', 'S']:
        sel = good & (d['PATCH'] == p)
        e1[sel] -= np.average(e1[sel], weights=w[sel])
        e2[sel] -= np.average(e2[sel], weights=w[sel])
    m = xc.mappers.MapperKiDSLegacy.m[zbin]
    return good, e1[good]/(1+m), e2[good]/(1+m), w[good]


def gen_catalog():
    d = get_data()
    cols = [fits.Column(name=k, format='8A' if k == 'PATCH' else 'D',
                        array=v) for k, v in d.items()]
    fits.BinTableHDU.from_columns(cols).writeto(CAT_PATH, overwrite=True)


def gen_nz():
    z = np.linspace(0, 2, 101)
    cols = [fits.Column(name='Z_LOW', format='D', array=z[:-1]),
            fits.Column(name='Z_MID', format='D',
                        array=0.5*(z[1:]+z[:-1])),
            fits.Column(name='Z_HIGH', format='D', array=z[1:])]
    cols += [fits.Column(name=f'BIN{i+1}', format='D',
                         array=(i+1)*np.ones(100)) for i in range(6)]
    hdu = fits.BinTableHDU.from_columns(cols, name='NZ_SOURCE')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(NZ_PATH, overwrite=True)


def remove_rerun(prerun):
    frerun = glob.glob(prerun + 'KiDSLegacy*.fits*')
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
            'mode': 'shear',
            'zbin': 0, 'nside': NSIDE, 'mask_name': 'mask',
            'coords': 'C'}


@pytest.fixture
def mapper(config):
    return xc.mappers.MapperKiDSLegacy(config)


def test_smoke(config):
    m = xc.mappers.MapperKiDSLegacy(config)
    assert m.map_name == 'KiDSLegacy_bin0'


@pytest.mark.parametrize("zbin", [6, -1])
def test_init_errors(config, zbin):
    config['zbin'] = zbin
    with pytest.raises(ValueError):
        xc.mappers.MapperKiDSLegacy(config)


@pytest.mark.parametrize("mode", ["PSF", "shear", "fail"])
def test_set_mode(mapper, mode):
    if mode == "PSF":
        assert mapper._set_mode(mode) == ('PSF_e1', 'PSF_e2', 'PSF')
    elif mode == "shear":
        assert mapper._set_mode(mode) == ('e1', 'e2', 'shear')
    else:
        with pytest.raises(ValueError):
            mapper._set_mode(mode)


@pytest.mark.parametrize("zbin", range(6))
def test_get_catalog(config, zbin):
    config['zbin'] = zbin
    m = xc.mappers.MapperKiDSLegacy(config)
    cat = m.get_catalog()
    good, e1, e2, w = get_expected(zbin)
    d = get_data()
    assert len(cat) == np.sum(good)
    assert np.all(cat['RA'] == d['RAJ2000'][good])
    assert np.all(cat['DEC'] == d['DECJ2000'][good])
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
    m = xc.mappers.MapperKiDSLegacy(config)
    z, nz = m.get_nz()
    zz = np.linspace(0, 2, 101)
    assert np.allclose(z, 0.5*(zz[1:]+zz[:-1]))
    assert np.all(nz == 3)
    z2, _ = m.get_nz(dz=0.1)
    assert np.allclose(z2, z+0.1)


def test_get_nmt_catalog_field(mapper):
    f = mapper._get_nmt_catalog_field()
    assert isinstance(f, xc.mappers.mapper_KiDSLegacy.nmt.NmtFieldCatalog)


def test_get_dtype(mapper):
    assert mapper.get_dtype() == 'galaxy_shear'


def test_get_spin(mapper):
    assert mapper.get_spin() == 2


def test_rerun(config):
    config['path_rerun'] = OUTDIR
    remove_rerun(OUTDIR)
    m = xc.mappers.MapperKiDSLegacy(config)
    s = m.get_signal_map()
    psf = m.get_signal_map('PSF')
    mask = m.get_mask()
    nl = m.get_nl_coupled()
    nl_psf = m.get_nl_coupled('PSF')

    # All bins of the catalog are saved
    for ib in range(6):
        assert os.path.isfile(os.path.join(OUTDIR,
                                           f'KiDSLegacy_bin{ib}_cat.fits'))
    for fname in [f'signal_map_shear_coordC_ns{NSIDE}.fits.gz',
                  f'signal_map_PSF_coordC_ns{NSIDE}.fits.gz',
                  f'shear_w2s2_coordC_ns{NSIDE}.fits.gz',
                  f'PSF_w2s2_coordC_ns{NSIDE}.fits.gz']:
        assert os.path.isfile(os.path.join(OUTDIR,
                                           f"KiDSLegacy_bin0_{fname}"))
    assert os.path.isfile(os.path.join(OUTDIR,
                                       f'mask_mask_coordC_ns{NSIDE}.fits.gz'))

    # Non-existing catalog - read from rerun
    config['data_catalog'] = 'whatever'
    m2 = xc.mappers.MapperKiDSLegacy(config)
    assert np.allclose(m.get_catalog()['e1'], m2.get_catalog()['e1'])
    assert np.all(np.array(s) == np.array(m2.get_signal_map()))
    assert np.all(np.array(psf) == np.array(m2.get_signal_map('PSF')))
    assert np.all(mask == m2.get_mask())
    # Allow for round-off differences when averaging maps read from file
    assert np.allclose(nl, m2.get_nl_coupled(), rtol=1E-10, atol=0)
    assert np.allclose(nl_psf, m2.get_nl_coupled('PSF'), rtol=1E-10, atol=0)

    # Other bins are read from rerun too
    config['zbin'] = 1
    m3 = xc.mappers.MapperKiDSLegacy(config)
    good, e1, _, _ = get_expected(1)
    assert np.allclose(m3.get_catalog()['e1'], e1)

    remove_rerun(OUTDIR)
