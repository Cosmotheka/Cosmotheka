from astropy.coordinates import SkyCoord
import numpy as np


def remove_further_duplicates(photo_index, dist_2d):
    """
    Remove duplicates matches. Keep only the closest galaxy.

    Arguments
    ---------
        photo_index (array): Array of indices of photometric galaxies with
        spectroscopic counterpart; i.e. pcat_xmat = photo_sample[photo_index]
        dist_2d  (Angle): The on-sky separation between the closest match for
        each element in this object in catalogcoord. Shape matches this object.

    Returns
    -------
        photo_index (array): Subset of the input photo_index array without
        duplicates.
        dist_2d (Angle): The corresponding dist_2d object.
        sel (array): The selection array to remove the duplicates; e.g.
        the returned photo_index = photo_index[sel]

    """
    pi_unique, count = np.unique(photo_index, return_counts=True)
    indices_repeated = pi_unique[count > 1]

    sel = np.array([True] * photo_index.size)
    for i in indices_repeated:
        ix_in_photo_index = np.where(photo_index == i)
        # Keep closest galaxy
        ix_to_keep = dist_2d[ix_in_photo_index].argmin()
        ix_to_delete = np.delete(ix_in_photo_index, ix_to_keep)
        sel[ix_to_delete] = False

    return photo_index[sel], dist_2d[sel], sel


def get_cross_match_gals(mapper1, mapper2, return_ix_xmat=False):
    """
    Match the galaxies in both cat1_sample and cat2_sample.

    Arguments
    ---------
        cat1 (fits): frist cataloge.
        cat2 (fits): second catalog.
        cat1_columns (list): List with the column names of the
        right ascension and declination columns for the first
        cataloge. They are asumed to be in degrees.
        cat2_columns (list): Same as photo_columns but for the second cataloge.
        return_ix_xmat (bool): If True return the indices that slice the
        photo_sample and spec_sample after cross-matching. Default False.
    Returns
    -------
        fits: photo_xmat: Subsample of the photometric sample that
        cross-matches with the spectroscopic
        fits: spec_xmat: As above, but for the spectroscopic sample
        array: pix_xmat: Array with the indices of the galaxies in the
        photometric sample with spectroscopic counterpart.
    """
    # Cut photo_sample around COSMOS area to speed up matching
    cat1 = mapper1.get_catalog()
    cat2 = mapper2.get_catalog()
    cat1_cols = mapper1._get_radec_names()
    cat2_cols = mapper2._get_radec_names()
    ra1, dec1 = np.array([np.array(cat1[i]) for i in cat1_cols])
    ra2, dec2 = np.array([np.array(cat2[i]) for i in cat2_cols]) 
    arcmin = 10/60
    sel = (ra1 >= ra2.min() - arcmin) * (ra1 <= ra2.max() + arcmin) * \
          (dec1 >= dec2.min() - arcmin) * (dec1 <= dec2.max() + arcmin)

    ra1 = ra1[sel]
    ra2 = ra2[sel]
    dec1 = dec1[sel]
    dec2 = dec2[sel]

    # Based on
    # https://github.com/LSSTDESC/DEHSC_LSS/blob/master/hsc_lss/cosmos_weight.py
    # Match coordinates
    cat1_skycoord = SkyCoord(ra=ra1, dec=dec1, unit='deg')
    cat2_skycoord = SkyCoord(ra=ra2, dec=dec2, unit='deg')

    # Nearest neighbors
    # Cross-match from spec to photo, not photo to spec
    cat1_index, dist_2d, _ = \
        cat2_skycoord.match_to_catalog_sky(cat1_skycoord)

    # Cut everything further than 1 arcsec
    mask = dist_2d.degree * 60 * 60 < 1
    pix_xmat = cat1_index[mask]
    cat1_xmat = cat1[pix_xmat]
    cat2_xmat = cat2[mask]

    # Check if there are multiple cross-matchings
    rdev = pix_xmat.size / np.unique(pix_xmat).size - 1
    print(f'Multiple cross-matching: {100 * rdev:.2f}%', flush=True)

    if np.abs(rdev) > 0:
        print('Removing multiple cross-matching', flush=True)
        pix_xmat, dist_2d_xmat, sel = remove_further_duplicates(pix_xmat,
                                                                dist_2d[mask])
        # Update mask
        ix_to_remove = np.where(~sel)[0]
        ix_true_in_mask = np.where(mask)[0]
        mask[ix_true_in_mask[ix_to_remove]] = False

        rdev = pix_xmat.size / np.unique(pix_xmat).size - 1
        print(f'Multiple cross-matching after cleaning: {100 * rdev:.2f}%',
              flush=True)

        cat1_xmat, cat2_xmat = cat1_xmat[sel], cat2_xmat[sel]

    if return_ix_xmat:
        return cat1_xmat, cat2_xmat, pix_xmat, mask
    else:
        return cat1_xmat, cat2_xmat
