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
    """Loads files from previous runs of a given mapper
    Args:
        mpr (:class:`~xcell.mapper_base.MapperBase`): A mapper object.
        fname (string): path to rerun files.
        ftype (string): type of rerun files FITSTable, FITSMap, ASCII or NPZ
        section (Int): if `ftype == FITSTable` or `ftype == FITSMap` selects \
            field of file.
        read (`True` or `False`): if `False` only checks for existance of \
            files as opposed to reading them.
    Returns:
        file
    """
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
    """Saves files from previous runs of a given mapper
    Args:
        mpr (:class:`~xcell.mapper_base.MapperBase`): A mapper object.
        fname (string): path to rerun files.
        ftype (string): type of rerun files FITSTable, FITSMap, ASCII or NPZ
        data: the data that will be saved to a file
    Returns:

    """
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


def rotate_mask(mask, rot, binarize=False):
    """Applies a given rotator object to mask
    Args:
        mask (Array): mask to be rotated.
        rot (:class:`healpy.rotator.Rotator`): rotator \
            object containing the current and target \
            coordinates.
        binarize(`True` or `False`): if `True` pixels \
            with values smaller than 0.5 are set to 0 \
            Pixels with values bigger than 0.5 are set \
            to 1.

    Returns:
        mask (Array)
    """
    if rot is None:
        return mask

    m = rot.rotate_map_pixel(mask)
    if binarize:
        m[m < 0.5] = 0
        m[m >= 0.5] = 1
    return m


def rotate_map(mapp, rot):
    """
    Applies a given rotator object to map. \
    The rotation is performed in Fourrier
    space.

    Args:
        - mapp (Array): map to be rotated.
        - rot (:class:`healpy.rotator.Rotator`): rotator \
               object containing the current and target \
               coordinates.

    Returns:
        map (Array)
    """
    if rot is None:
        return mapp
    return rot.rotate_map_alms(mapp)


def get_map_from_points(cat, nside, w=None, rot=None,
                        ra_name='RA', dec_name='DEC',
                        in_radians=False, qu=None):
    """
    Creates a map given a catalog of objects and \
    a number of pixels.

    Args:
        cat (Array): catalog of sources.
        nside (:class:`healpy.rotator.Rotator`): rotator \
            object containing the current and target \
            coordinates.
        w (Array): weights of sources in catalaog. \
                   Defaults to `None`.
       rot (:class:`healpy.rotator.Rotator`): If not `None` \
           rotates sources in catalog to target \
           coordinates. Defaults to "None". \
       ra_name (String): name of RA field in catalog. \
                         Defaults to "RA".
       dec_name (String): name of DEC field in catalog. \
                          Defaults to "DEC".
       in_radians (`True` or `False`): Flags if input catalog. \
           If `True`, converts angles in catalog to degrees.
       qu (Array): weights for spin-2 quantities. \
                   Defaults to `None`.

    Returns:
        numcount (Array)
    """
    npix = hp.nside2npix(nside)
    if in_radians:
        lon = np.degrees(cat[ra_name])
        lat = np.degrees(cat[dec_name])
    else:
        lon = cat[ra_name]
        lat = cat[dec_name]
    if rot is not None:
        # Rotate spin-2 quantities if needed
        if qu is not None:
            angle_ref = rot.angle_ref(lon, lat, lonlat=True)
            ll = (qu[0] + 1j*qu[1])*np.exp(1j*2*angle_ref)
            qu = [np.real(ll), np.imag(ll)]
        # Rotate coordinates
        lon, lat = rot(lon, lat, lonlat=True)
    ipix = hp.ang2pix(nside, lon, lat, lonlat=True)
    if qu is not None:
        if w is not None:
            qu = [x*w for x in qu]
        q = np.bincount(ipix, weights=qu[0], minlength=npix)
        u = np.bincount(ipix, weights=qu[1], minlength=npix)
        numcount = [q, u]
    else:
        numcount = np.bincount(ipix, weights=w, minlength=npix)
    return numcount


def get_DIR_Nz(cat_spec, cat_photo, bands, zflag,
               zrange, nz, nearest_neighbors=10, njk=100,
               bands_photo=None):
    """
    Implementation of the DIR algorithm that \
    calibrates a photometric galaxy redshift \
    distribution given a reference \
    spectroscopic sample. \
    The code makes use of sklearn nearest \
    neighbors algorthim

    Args:
        - cat_spec (Array): catalog of spectroscopic \
                            samples.
        - cat_photo (Array): catalog of photometric \
                             samples.
        - bands (Array): bands for spectroscopic \
                         and photometric samples.
        - zflag (String): name of the redshift field \
                        in the samples catalog.
        - zrange (Array): redshift range for the calibrated \
           galaxy redshift distribution.
        - nz (Int): number of bins for calibrated \
          galaxy redshift distribution.
        - nearest_neighbors = 10 (float): number of nearest neighbors \
           found by the algorithm.
        - njk (float) = 100: Loop over JK region.
        - bands_photo = None: bands of phometric catalog.

    Returns:
        - zz (Array): position of bins in redshift
        - dndz (Array): elements per bin
        - dndz_jk (Array):
    """
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
