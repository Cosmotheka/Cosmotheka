import xcell as xc
import numpy as np
import healpy as hp
from astropy.table import Table
import os


def get_config():
    c = {'exposure_map': 'xcell/tests/data/exp_rosat.fits',
         'photon_list': 'xcell/tests/data/cat_rosat.fits',
         'external_mask': 'xcell/tests/data/msk_rosat.fits',
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
    cat.write('xcell/tests/data/cat_rosat.fits', overwrite=True)
    hp.write_map('xcell/tests/data/exp_rosat.fits', exp, overwrite=True)
    hp.write_map('xcell/tests/data/msk_rosat.fits', msk, overwrite=True)


def clean_rosat_data():
    for fn in ['cat_rosat.fits', 'msk_rosat.fits', 'exp_rosat.fits']:
        fname = f'xcell/tests/data/{fn}'
        if os.path.isfile(fname):
            os.remove(fname)


def test_smoke():
    make_rosat_data()
    c = get_config()
    m = xc.mappers.MapperROSATXray(c)
    assert m.get_spin() == 0
    assert m.get_dtype() == 'generic'
    cat = m.get_pholist()
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
    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    mp = m.get_signal_map()[0]
    apix = hp.nside2pixarea(m.nside)
    assert np.all(np.fabs(mp - 1./200./apix) < 1E-5)
    clean_rosat_data()


def test_pholist():
    c = get_config()
    nside = c['nside']
    npix = hp.nside2npix(nside)

    make_rosat_data()
    m = xc.mappers.MapperROSATXray(c)
    cat = m.get_pholist()
    assert len(cat) < npix
    clean_rosat_data()

    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    cat = m.get_pholist()
    assert len(cat) == npix
    clean_rosat_data()


def test_nl_coupled():
    c = get_config()
    make_rosat_data(e0=1., ef=1.5)
    m = xc.mappers.MapperROSATXray(c)
    nl = m.get_nl_coupled()
    apix = hp.nside2pixarea(c['nside'])
    nl_guess = 1/(200.**2*apix)
    assert np.all(np.fabs(nl/nl_guess-1) < 1E-10)
    clean_rosat_data()
