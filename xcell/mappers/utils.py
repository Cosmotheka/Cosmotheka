import numpy as np
import healpy as hp
import fitsio
import os


def _build_rerun_fname(mpr, fname):
    # Check if we want to save rerun data
    path = mpr.config.get('path_rerun', None)
    if path is None:
        return None, False

    os.makedirs(path, exist_ok=True)

    # Check if file exists
    fname_full = os.path.join(path, fname)
    return fname_full, os.path.isfile(fname_full)


def get_rerun_data(mpr, fname, ftype, section=None, read=True):
    # Ignore rerun file if required
    ignore = mpr.config.get('ignore_rerun', False)
    if ignore:
        return None

    fname_full, exists = _build_rerun_fname(mpr, fname)

    # Check if we want to save rerun data
    if fname_full is None:
        return None

    # If just checking for existence, return True/False
    if not read:
        return exists

    # If it doesn't exist, just return False
    if not exists:
        return None

    # Read
    if ftype == 'FITSTable':
        return fitsio.read(fname_full, ext=section)
    elif ftype == 'FITSMap':
        return np.array(hp.read_map(fname_full, field=section))
    elif ftype == 'ASCII':
        return np.loadtxt(fname_full, unpack=True)
    elif ftype == 'NPZ':
        d = np.load(fname_full)
        return dict(d)
    else:
        raise ValueError(f"Unknown file format {ftype}")


def save_rerun_data(mpr, fname, ftype, data):
    fname_full, _ = _build_rerun_fname(mpr, fname)

    if fname_full is None:
        return

    if ftype == 'FITSTable':
        fitsio.write(fname_full, data, clobber=True)
    elif ftype == 'FITSMap':
        hp.write_map(fname_full, data, overwrite=True)
    elif ftype == 'ASCII':
        np.savetxt(fname_full, data)
    elif ftype == 'NPZ':
        np.savez(fname_full, **data)
    else:
        raise ValueError(f"Unknown file format {ftype}")


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
