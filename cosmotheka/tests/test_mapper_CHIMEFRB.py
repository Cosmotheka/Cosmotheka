import numpy as np
import cosmotheka as xc
import healpy as hp
import fitsio
import os
import shutil
import pytest


OUTDIR = 'cosmotheka/tests/data/tmp_chimefrb/'
CAT_PATH = OUTDIR + 'chimefrbcat.fits'
NSIDE = 32
NPIX = hp.nside2npix(NSIDE)


def get_data():
    # One burst per pixel
    ipix = np.arange(NPIX)
    ra, dec = hp.pix2ang(NSIDE, ipix, lonlat=True)
    gl, gb = hp.Rotator(coord=['C', 'G'])(ra, dec, lonlat=True)
    dm = 100. + (ipix % 10).astype(float)
    sub_num = np.zeros(NPIX, dtype=int)
    sub_num[ipix % 11 == 0] = 1
    excluded = (ipix % 13 == 0).astype(int)
    sidelobe = (ipix % 17 == 0).astype(int)
    # Two repeaters, each with 3 bursts
    rep = np.array(['' for i in ipix], dtype='U16')
    rep[[1, 2, 3]] = 'FRB_R1'
    rep[[4, 5, 6]] = 'FRB_R2'
    mjd = 58000. + 100.*(ipix % 7)
    ra[0] = np.nan
    dec[0] = np.nan
    d = np.zeros(NPIX, dtype=[('ra', 'f8'), ('dec', 'f8'),
                              ('gl', 'f8'), ('gb', 'f8'),
                              ('dm_fitb', 'f8'),
                              ('dm_exc_ne2001', 'f8'),
                              ('dm_exc_ymw16', 'f8'),
                              ('sub_num', 'i8'), ('excluded_flag', 'i8'),
                              ('sidelobe_flag', 'i8'),
                              ('repeater_name', 'U16'),
                              ('mjd_400', 'f8')])
    d['ra'] = ra
    d['dec'] = dec
    d['gl'] = gl
    d['gb'] = gb
    # Different Milky Way contributions for each model
    d['dm_fitb'] = dm + 50.
    # Failed fits (including one repeater burst)
    d['dm_fitb'][ipix % 19 == 0] = np.nan
    d['dm_fitb'][2] = np.nan
    d['dm_exc_ne2001'] = dm
    d['dm_exc_ymw16'] = dm + 10.*(ipix % 3)
    d['sub_num'] = sub_num
    d['excluded_flag'] = excluded
    d['sidelobe_flag'] = sidelobe
    d['repeater_name'] = rep
    d['mjd_400'] = mjd
    return d


def get_expected(repeaters='average', dm_col='dm_exc_ne2001'):
    # Independent calculation of the selected bursts. Returns the
    # pixel of each selected entry and its DM.
    d = get_data()
    ipix = np.arange(NPIX)
    good = ((d['sub_num'] == 0) & (d['excluded_flag'] == 0) &
            (d['sidelobe_flag'] == 0) & np.isfinite(d['ra']) &
            np.isfinite(d['dm_fitb']))
    is_rep = d['repeater_name'] != ''
    sel = good & ~is_rep
    pix = list(ipix[sel])
    dm = list(d[dm_col][sel])
    if repeaters == 'all':
        sel = good & is_rep
        pix += list(ipix[sel])
        dm += list(d[dm_col][sel])
    elif repeaters == 'average':
        for name in ['FRB_R1', 'FRB_R2']:
            sel = good & (d['repeater_name'] == name)
            ids = ipix[sel]
            pix.append(ids[np.argmin(d['mjd_400'][ids])])
            dm.append(np.mean(d[dm_col][ids]))
    return np.array(pix), np.array(dm)


def gen_catalog():
    fitsio.write(CAT_PATH, get_data(), clobber=True)


def setup_module():
    os.makedirs(OUTDIR, exist_ok=True)
    gen_catalog()


def teardown_module():
    shutil.rmtree(OUTDIR, ignore_errors=True)


@pytest.fixture
def config():
    return {'data_catalog': CAT_PATH,
            'nside': NSIDE, 'mask_name': 'mask',
            'coords': 'C'}


@pytest.fixture
def mapper(config):
    return xc.mappers.MapperCHIMEFRB(config)


def test_smoke(mapper):
    assert mapper.map_name == 'CHIMEFRB_MWne2001_repaverage'
    assert mapper.get_spin() == 0
    assert mapper.get_dtype() == 'generic'


@pytest.mark.parametrize("key", ['repeaters', 'galactic_model'])
def test_init_errors(config, key):
    config[key] = 'fail'
    with pytest.raises(ValueError):
        xc.mappers.MapperCHIMEFRB(config)


@pytest.mark.parametrize("repeaters", ['average', 'all', 'exclude'])
def test_get_catalog(config, repeaters):
    config['repeaters'] = repeaters
    m = xc.mappers.MapperCHIMEFRB(config)
    cat = m.get_catalog()
    pix, dm = get_expected(repeaters)
    ipix = hp.ang2pix(NSIDE, cat['RA'], cat['DEC'], lonlat=True)
    assert len(cat) == len(pix)
    assert np.all(ipix == pix)
    assert np.allclose(cat['DM'], dm)
    assert np.isclose(m.get_mean_dm(), np.mean(dm))


@pytest.mark.parametrize("model,dm_col", [('ne2001', 'dm_exc_ne2001'),
                                          ('ymw16', 'dm_exc_ymw16'),
                                          (None, 'dm_fitb'),
                                          ('none', 'dm_fitb')])
def test_galactic_model(config, model, dm_col):
    config['galactic_model'] = model
    m = xc.mappers.MapperCHIMEFRB(config)
    assert m.dm_column == dm_col
    pix, dm = get_expected(dm_col=dm_col)
    cat = m.get_catalog()
    assert np.allclose(cat['DM'], dm)


def test_get_catalog_cuts(config):
    config['dm_range'] = [102, 106]
    config['gb_min'] = 20.
    m = xc.mappers.MapperCHIMEFRB(config)
    cat = m.get_catalog()
    _, gb = hp.Rotator(coord=['C', 'G'])(cat['RA'], cat['DEC'],
                                         lonlat=True)
    assert np.all(np.fabs(gb) >= 20.)
    assert np.all((cat['DM'] >= 102) & (cat['DM'] < 106))


def test_get_mask(mapper):
    pix, _ = get_expected()
    mask = mapper.get_mask()
    assert len(mask) == NPIX
    assert np.all(mask[pix] == 1)
    assert np.sum(mask) == len(pix)


def test_get_signal_map(mapper):
    pix, dm = get_expected()
    mp = mapper.get_signal_map()
    assert mp.shape == (1, NPIX)
    assert np.allclose(mp[0][pix], dm - np.mean(dm))
    assert np.sum(mp[0] != 0) <= len(pix)
    # Check pre-loaded map
    assert np.all(mapper.get_signal_map() == mp)


def test_get_nl_coupled(mapper):
    pix, dm = get_expected()
    nlp = hp.nside2pixarea(NSIDE)*np.sum((dm-np.mean(dm))**2)/NPIX
    nl = mapper.get_nl_coupled()
    assert nl.shape == (1, 3*NSIDE)
    assert np.allclose(nl, nlp)


@pytest.mark.parametrize("coords", ['C', 'G'])
def test_get_nmt_catalog_field(config, coords):
    config['coords'] = coords
    m = xc.mappers.MapperCHIMEFRB(config)
    f = m._get_nmt_catalog_field()
    assert f.spin == 0
