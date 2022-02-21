import numpy as np
from xcell.mappers import MapperDummy
from astropy.coordinates import SkyCoord
from xcell.cls.utils import remove_further_duplicates, get_cross_match_gals


def get_config(fsky=0.2, fsky2=0.3,
               dtype0='galaxy_density',
               dtype1='galaxy_density',
               inc_hm=False):
    nside = 32
    # Set only the necessary entries. Leave the others to their default value.
    cosmo = {
        # Planck 2018: Table 2 of 1807.06209
        # Omega_m: 0.3133
        'Omega_c': 0.2640,
        'Omega_b': 0.0493,
        'h': 0.6736,
        'n_s': 0.9649,
        'sigma8': 0.8111,
        'w0': -1,
        'wa': 0,
        'transfer_function': 'eisenstein_hu',
        'baryons_power_spectrum': 'nobaryons',
    }
    dummy0 = {'mask_name': 'mask_dummy2', 'mapper_class': 'MapperDummy',
              'cosmo': cosmo, 'nside': nside, 'fsky': fsky2, 'seed': 100,
              'dtype': dtype1, 'use_halo_model': inc_hm, 'mask_power': 2,
              'ra0': 60, 'dec0': 60,
              'catalog': 'xcell/tests/data/catalog.fits'}
    return {'tracers': {'Dummy__0': dummy0}}


def test_get_cross_match_gals():
    data = get_config()
    mapper = MapperDummy(data['tracers']['Dummy__0'])
    gals12, gals21 = get_cross_match_gals(mapper, mapper)
    assert len(gals12) == len(gals21)
    assert len(gals12) == len(mapper.get_catalog())


def test_remove_further_duplicates():
    data = get_config()
    mapper = MapperDummy(data['tracers']['Dummy__0'])
    cat = mapper.get_catalog()
    cat_cols = mapper._get_radec_names()
    ra, dec = np.array([np.array(cat[i]) for i in cat_cols])
    cat1_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')
    cat2_skycoord = SkyCoord(ra=ra, dec=dec, unit='deg')
    cat1_index, dist_2d, _ = \
        cat2_skycoord.match_to_catalog_sky(cat1_skycoord)
    mask = dist_2d.degree * 60 * 60 < 1
    pix_xmat = cat1_index[mask]
    pix_xmat, dist_2d_xmat, sel = remove_further_duplicates(pix_xmat,
                                                            dist_2d[mask])
    assert remove_further_duplicates(pix_xmat, dist_2d[mask])
