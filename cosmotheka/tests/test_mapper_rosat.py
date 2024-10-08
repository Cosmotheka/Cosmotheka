import cosmotheka as xc
import numpy as np
import healpy as hp
from astropy.table import Table
import os


def get_config(mask_ps=False):
    c = {'exposure_map': 'cosmotheka/tests/data/exp_rosat.fits',
         'photon_list': 'cosmotheka/tests/data/cat_rosat.fits',
         'external_mask': 'cosmotheka/tests/data/msk_rosat.fits',
         'mask_point_sources': mask_ps,
         'point_source_catalog': 'cosmotheka/tests/data/cat_ps_rosat.fits',
         'point_source_flux_cut_ctps': 0.02,
         'energy_range': [0.5, 3.0],
         'exposure_min': 100.,
         'mask_name': 'mask_ROSAT',
         'nside': 32, 'coords': 'C'}
    return c


def make_rosat_data(e0=0., ef=4., expval=200.):
    nside = 32
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    ener = e0 + (ef-e0)*np.random.rand(npix)
    exp = np.ones(npix)*expval
    msk = np.ones(npix)
    cat = Table({'raj2000': ra,
                 'dej2000': dec,
                 'energy_cor': ener,
                 'exposure_time': exp})
    cat.write('cosmotheka/tests/data/cat_rosat.fits', overwrite=True)
    cat = Table({'RA_DEG': ra[:10],
                 'DEC_DEG': dec[:10],
                 'RATE': np.ones(10)})
    cat.write('cosmotheka/tests/data/cat_ps_rosat.fits', overwrite=True)
    hp.write_map('cosmotheka/tests/data/exp_rosat.fits', exp, overwrite=True)
    hp.write_map('cosmotheka/tests/data/msk_rosat.fits', msk, overwrite=True)


def clean_rosat_data():
    for fn in ['cat_rosat.fits', 'cat_ps_rosat.fits',
               'msk_rosat.fits', 'exp_rosat.fits']:
        fname = f'cosmotheka/tests/data/{fn}'
        if os.path.isfile(fname):
            os.remove(fname)


def test_smoke():
    make_rosat_data()
    c = get_config()
    m = xc.mappers.MapperROSATXray(c)
    assert m.get_spin() == 0
    assert m.get_dtype() == 'generic'
    cat = m._get_pholist()
    assert np.amax(cat['energy_cor']) <= c['energy_range'][1]
    assert np.amin(cat['energy_cor']) >= c['energy_range'][0]
    clean_rosat_data()


def test_mask_sanity():
    make_rosat_data(expval=10.)
    c = get_config()
    m = xc.mappers.MapperROSATXray(c)
    msk = m.get_mask()
    assert np.all(msk == 0)
    clean_rosat_data()


def test_map_sanity():
    c = get_config()
    c['path_rerun'] = 'cosmotheka/tests/data/'
    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    mp = m.get_signal_map()[0]
    apix = hp.nside2pixarea(m.nside)
    assert np.all(np.fabs(mp - 1./200./apix) < 1E-5)

    fn = 'cosmotheka/tests/data/ROSATXray_signal_map_coordC_ns32.fits.gz'
    assert np.all(mp == hp.read_map(fn))
    os.remove(fn)
    clean_rosat_data()


def test_map_sanity_masked():
    c = get_config(True)
    c['path_rerun'] = 'cosmotheka/tests/data/'
    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    mp = m.get_signal_map()[0]
    apix = hp.nside2pixarea(m.nside)
    val = 1./200./apix
    mapval = np.unique(mp)
    assert len(mapval) == 2
    assert np.amin(mapval) == 0
    assert np.fabs(np.amax(mapval)-val) < 1E-5

    fn = 'cosmotheka/tests/data/ROSATXray_signal_map_coordC_ns32.fits.gz'
    assert np.all(mp == hp.read_map(fn))
    os.remove(fn)
    os.remove('cosmotheka/tests/data/mask_mask_ROSAT_coordC_ns32.fits.gz')
    clean_rosat_data()


def test_pholist():
    c = get_config()
    nside = c['nside']
    npix = hp.nside2npix(nside)

    make_rosat_data()
    m = xc.mappers.MapperROSATXray(c)
    cat = m._get_pholist()
    assert len(cat) < npix
    clean_rosat_data()

    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    cat = m._get_pholist()
    assert len(cat) == npix
    clean_rosat_data()


def test_nl_coupled():
    c = get_config()
    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    nl = m.get_nl_coupled()
    apix = hp.nside2pixarea(c['nside'])
    nl_guess = 1/apix
    assert np.all(np.fabs(nl/nl_guess-1) < 1E-10)
    clean_rosat_data()
