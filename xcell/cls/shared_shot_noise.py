def cross_match_gals(cat1, cat2, cat1_columns,
                     cat2_columns, return_ix_xmat=False):
    """
    Match the galaxies in both cat1_sample and cat2_sample.

    Arguments
    ---------
        cat1 (fits): frist cataloge.
        cat2 (fits): second catalog.
        cat1_columns (list): List with the column names of the right ascension
        and declination columns for the first cataloge. They are asumed to be in
        degrees.
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
    ra1, dec1 = columns_from_fits(cat1, cat1_columns)
    ra2, dec2 = columns_from_fits(cat2, cat2_columns)
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
    cat1_xmat = cat1_sample[pix_xmat]
    cat2_xmat = cat2_sample[mask]

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

    
def get_shared_shot_noise(mapper1, mapper2):
    cat1 = mapper1.get_catalog()
    cat2 = mapper2.get_catalog()
    cols1 = mapper1._get_cat_cols()
    cols2 = mapper2._get_cat_cols()
    shared_cat = cross_match_gals(cat1, cat2, cols1, cols2)
    shared_count = len(shared_cat)
    if shared_count == 0:
        shot_noise = 0 
    else: 
        cat1_count = len(cat1)
        cat2_count = len(cat2)
        shot_noise = (shared_count/(cat1_count+cat2_count))
    return shot_noise
    