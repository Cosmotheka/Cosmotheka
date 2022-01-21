import numpy as np
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d


def get_map_from_points(cat, nside, w=None,
                        ra_name='RA', dec_name='DEC',
                        in_radians=False):
    npix = hp.nside2npix(nside)
    if in_radians:
        ipix = hp.ang2pix(nside,
                          np.degrees(cat[ra_name]),
                          np.degrees(cat[dec_name]),
                          lonlat=True)
    else:
        ipix = hp.ang2pix(nside, cat[ra_name], cat[dec_name],
                          lonlat=True)
    numcount = np.bincount(ipix, weights=w, minlength=npix)
    return numcount


def get_DIR_Nz(cat_spec, cat_photo, bands, zflag,
               zrange, nz, nearest_neighbors=10, njk=100,
               bands_photo=None):
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial import cKDTree
    train_data = np.array([cat_spec[c] for c in bands]).T
    if bands_photo is None:
        bands_photo = bands
    photo_data = np.array([cat_photo[c] for c in bands_photo]).T

    # Get nearest neighbors
    n_nbrs = NearestNeighbors(n_neighbors=nearest_neighbors,
                              algorithm='kd_tree',
                              metric='euclidean').fit(train_data)
    # Get distances
    distances, _ = n_nbrs.kneighbors(train_data)
    # Get maximum distance
    distances = np.amax(distances, axis=1)
    # Find all photo-z objects within this
    # distance of each spec-z object
    tree_NN_lookup = cKDTree(photo_data, leafsize=40)
    num_photoz = np.array([len(tree_NN_lookup.query_ball_point(t, d+1E-6))
                           for t, d in zip(train_data, distances)])
    # Weight as ratio of numbers
    ntrain = len(train_data)
    weights = (num_photoz * ntrain /
               (nearest_neighbors * len(photo_data)))

    # Compute N(z)
    dndz, zz = np.histogram(cat_spec[zflag], range=zrange, bins=nz,
                            weights=weights, density=True)
    # Loop over JK regions
    dndz_jk = []
    for i in range(njk):
        id0 = int(ntrain*i/njk)
        idf = int(ntrain*(i+1)/njk)
        msk = (np.arange(ntrain) >= idf) | (np.arange(ntrain) < id0)
        n, _ = np.histogram(cat_spec[zflag][msk], range=zrange,
                            bins=nz, weights=weights[msk], density=True)
        dndz_jk.append(n)
    dndz_jk = np.array(dndz_jk)
    return zz, dndz, dndz_jk


def _beam_gaussian(ell, fwhm_amin):
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)


def get_beam(nside, beam_info):
    if beam_info is None:  # No beam
        beam = np.ones(3*nside)
    elif beam_info['type'] == 'Gaussian':  # Gaussian beam
        ell = np.arange(3*nside)
        beam = _beam_gaussian(ell, beam_info['FWHM_arcmin'])
        beam /= beam[0]  # normalize it
    else:
        raise NotImplementedError("Unknown beam type.")
    return beam


def _get_custom_wf(wf_info):
    field = wf_info['field']
    file = wf_info['file']
    windowfuncs = pd.read_csv(file, comment='#')
    wf = interp1d(np.array(windowfuncs['ell']),
                  np.array(windowfuncs[field]),
                  fill_value='extrapolate')
    return wf


def _get_pixel_wf(wf_info):
    nside_native = wf_info['nside_native']
    nside_wanted = wf_info['nside_wanted']
    ell_native = np.arange(3*nside_native)
    ell_wanted = np.arange(3*nside_wanted)
    wf_native = interp1d(ell_native,
                         hp.sphtfunc.pixwin(nside_native)[ell_native],
                         fill_value='extrapolate')
    wf_wanted = interp1d(ell_wanted,
                         hp.sphtfunc.pixwin(nside_wanted)[ell_wanted],
                         fill_value='extrapolate')
    wf = interp1d(ell_wanted,
                  wf_native(ell_wanted)/wf_wanted(ell_wanted),
                  fill_value='extrapolate')
    return wf


def get_wf(nside, wf_infos):
    ell = np.arange(3*nside)
    wf = 1.0*np.ones_like(ell)
    for wf_info in wf_infos:
        if wf_info is None:
            raise NotImplementedError("No window function info")
        elif wf_info['type'] == 'Pixel':
            wf *= _get_pixel_wf(wf_info)(ell)
        elif wf_info['type'] == 'Custom':
            wf *= _get_custom_wf(wf_info)(ell)
        else:
            raise NotImplementedError("Unknown settings.")

    wf = interp1d(ell, wf,
                  fill_value='extrapolate')

    return wf
