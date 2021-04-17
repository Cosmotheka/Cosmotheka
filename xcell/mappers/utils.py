import numpy as np
import healpy as hp


def get_map_from_points(cat, nside, w=None,
                        ra_name='RA', dec_name='DEC'):
    npix = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, cat[ra_name], cat[dec_name],
                      lonlat=True)
    numcount = np.bincount(ipix, weights=w, minlength=npix)
    return numcount


def get_DIR_Nz(cat_spec, cat_photo, bands, zflag,
               zrange, nz, nearest_neighbors=10):
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial import cKDTree
    train_data = np.array([cat_spec[c] for c in bands]).T
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
    photo_data = np.array([cat_photo[c] for c in bands]).T
    tree_NN_lookup = cKDTree(photo_data, leafsize=40)
    num_photoz = np.array([len(tree_NN_lookup.query_ball_point(t, d+1E-6))
                           for t, d in zip(train_data, distances)])
    # Weight as ratio of numbers
    weights = (num_photoz * len(train_data) /
               (nearest_neighbors * len(photo_data)))
    # Compute N(z)
    dndz, zz = np.histogram(cat_spec[zflag], range=zrange, bins=nz,
                            weights=weights, density=True)
    return zz, dndz
