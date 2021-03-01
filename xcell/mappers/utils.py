import numpy as np
import healpy as hp


def get_map_from_points(cat, nside, w=None,
                        ra_name='RA', dec_name='DEC'):
    npix = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, cat[ra_name], cat[dec_name],
                      lonlat=True)
    numcount = np.bincount(ipix, weights=w, minlength=npix)
    return numcount
