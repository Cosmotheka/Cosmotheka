from astropy.coordinates import SkyCoord
import numpy as np


def remove_further_duplicates(index, dist_2d):

    pi_unique, count = np.unique(index, return_counts=True)
    indices_repeated = pi_unique[count > 1]

    sel = np.array([True] * index.size)
    for i in indices_repeated:
        ix_in_index = np.where(index == i)
        # Keep closest galaxy
        ix_to_keep = dist_2d[ix_in_index].argmin()
        ix_to_delete = np.delete(ix_in_index, ix_to_keep)
        sel[ix_to_delete] = False

    return index[sel], dist_2d[sel], sel


def get_cross_match_gals(mapper1, mapper2, return_ix_xmat=False):
    # Cut photo_sample around COSMOS area to speed up matching
    cat1 = mapper1.get_catalog()
    cat2 = mapper2.get_catalog()
    ra1, dec1 = mapper1.get_radec()
    ra2, dec2 = mapper2.get_radec()

    # No significant improvment in computing time and can
    # lead to problems for masks that wrap around the sky
    # arcmin = 10/60
    # sel = (ra1 >= ra2.min() - arcmin) * (ra1 <= ra2.max() + arcmin) * \
    #       (dec1 >= dec2.min() - arcmin) * (dec1 <= dec2.max() + arcmin)
    # ra1 = ra1[sel]
    # ra2 = ra2[sel]
    # dec1 = dec1[sel]
    # dec2 = dec2[sel]

    cat1_skycoord = SkyCoord(ra=ra1, dec=dec1, unit='deg')
    cat2_skycoord = SkyCoord(ra=ra2, dec=dec2, unit='deg')

    # Nearest neighbors
    # Cross-match from spec to photo, not photo to spec
    cat1_index, dist_2d, _ = \
        cat2_skycoord.match_to_catalog_sky(cat1_skycoord)

    # Cut everything further than 1 arcsec
    mask = dist_2d.degree * 60 * 60 < 1
    id_xmat = cat1_index[mask]
    cat1_xmat = cat1[id_xmat]
    # Here cat2_xmat is not really interesting because
    # it has the same length as cat1_xmat by construction
    cat2_xmat = cat2[mask]

    if len(cat1_xmat) == len(cat2_xmat):
        cat_xmat = cat1_xmat
    else:
        raise NotImplementedError("len(cat1_xmat) != len(cat2_xmat)")

    # Check if there are multiple cross-matchings
    rdev = id_xmat.size / np.unique(id_xmat).size - 1
    print(f'Multiple cross-matching: {100 * rdev:.2f}%', flush=True)

    if np.abs(rdev) > 0:
        print('Removing multiple cross-matching', flush=True)
        pix_xmat, dist_2d_xmat, sel = remove_further_duplicates(id_xmat,
                                                                dist_2d[mask])
        # Update mask
        ix_to_remove = np.where(~sel)[0]
        ix_true_in_mask = np.where(mask)[0]
        mask[ix_true_in_mask[ix_to_remove]] = False

        rdev = pix_xmat.size / np.unique(pix_xmat).size - 1
        print(f'Multiple cross-matching after cleaning: {100 * rdev:.2f}%',
              flush=True)

        cat_xmat = cat_xmat[sel]

    if return_ix_xmat:
        return cat_xmat, pix_xmat, mask
    else:
        return cat_xmat
