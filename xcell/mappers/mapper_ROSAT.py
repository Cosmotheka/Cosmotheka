from .utils import get_map_from_points, rotate_map, rotate_mask
from .mapper_base import MapperBase
import fitsio
import numpy as np
import healpy as hp


class MapperROSATXray(MapperBase):
    """ Implements X-ray count rate maps from ROSAT.

    A few details so we can document this properly in the future:

    Photon list was obtained from:
    http://dc.zah.uni-heidelberg.de/tableinfo/rosat.photons

    Exposure maps were obtained from e.g.:
    https://heasarc.gsfc.nasa.gov/FTP/rosat/data/\
    pspc/processed_data/rass/release/rs931844n00/

    The signal map in this case is the count rate density in
    units of counts/second/sr^-1. The model for this would
    need the effective instrument area. For ROSAT this is an
    energy-dependent quantity which includes a transfer function
    for E_true vs. E_channel. These can be found in
    https://heasarc.gsfc.nasa.gov/docs/rosat/pspc_matrices.html
    but we don't need them and won't provide them here.

    The mask is constructed by thresholding the exposure map
    and combining it with any other externa map (e.g. aiming
    to remove Galactic emission).
    """
    map_name = 'ROSATXray'

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.fname_expmap = config['exposure_map']
        self.fname_pholist = config['photon_list']
        self.erange = config.get('energy_range', [0.5, 3.0])
        self.explimit = config.get('exposure_min', 100.0)
        self.mask_external = config.get('external_mask', None)
        self.npix = hp.nside2npix(self.nside)

        self.expmap = None
        self.pholist = None
        # signal_map is a map of the countrate
        self.nl_coupled = None

    def get_pholist(self):
        if self.pholist is None:
            f = fitsio.FITS(self.fname_pholist)
            cat = f[1].read()
            msk = ((cat['energy_cor'] < self.erange[1]) &
                   (cat['energy_cor'] > self.erange[0]))
            self.pholist = cat[msk]
        return self.pholist

    def get_expmap(self):
        if self.expmap is None:
            mp = hp.read_map(self.fname_expmap)
            mp = rotate_map(mp, self.rot)
            self.expmap = hp.ud_grade(mp, nside_out=self.nside)
        return self.expmap

    def _get_signal_map(self):
        cat = self.get_pholist()
        xpmap = self.get_expmap()
        mask = self.get_mask()
        count_map = get_map_from_points(cat, self.nside,
                                        ra_name='raj2000',
                                        dec_name='dej2000',
                                        rot=self.rot)
        signal_map = np.zeros(self.npix)
        goodpix = mask > 0.0
        signal_map[goodpix] = count_map[goodpix] / xpmap[goodpix]
        pixA = hp.nside2pixarea(self.nside)
        signal_map *= 1/pixA
        signal_map = np.array([signal_map])
        return signal_map

    def _get_mask(self):
        mask = np.ones(self.npix)
        xpmap = self.get_expmap()
        mask[xpmap <= self.explimit] = 0
        if self.mask_external is not None:
            msk = hp.read_map(self.mask_external)
            msk = rotate_mask(msk, self.rot, binarize=True)
            msk = hp.ud_grade(msk, nside_out=self.nside)
            mask *= msk
        return mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat = self.get_pholist()
            xpmap = self.get_expmap()
            mask = self.get_mask()
            count_map = get_map_from_points(cat, self.nside,
                                            ra_name='raj2000',
                                            dec_name='dej2000',
                                            rot=self.rot)
            goodpix = mask > 0.0
            # Mean count rate
            # CR_mean = \sum n_p / \sum exp_p
            cr_mean = np.sum(count_map[goodpix])/np.sum(xpmap[goodpix])
            # <mask^2/exposure>
            m2_ie = np.sum(mask[goodpix]**2/xpmap[goodpix]) / self.npix
            # Pixel area
            pixA = hp.nside2pixarea(self.nside)
            N_mean = cr_mean * m2_ie / pixA
            self.nl_coupled = N_mean * np.ones([1, 3*self.nside])
        return self.nl_coupled

    def get_dtype(self):
        return 'generic'

    def get_spin(self):
        return 0
