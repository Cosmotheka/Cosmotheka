from .mapper_base import MapperBase
from .utils import get_map_from_points
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
    def __init__(self, config):
        self._get_defaults(config)
        self.fname_expmap = config['exposure_map']
        self.fname_pholist = config['photon_list']
        self.erange = config.get('energy_range', [0.5, 3.0])
        self.explimit = config.get('exposure_min', 100.0)
        self.mask_external = config.get('external_mask', None)
        self.npix = hp.nside2npix(self.nside)

        self.expmap = None
        self.pholist = None
        self.countrate_map = None
        self.mask = None

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
            self.expmap = hp.ud_grade(mp, nside_out=self.nside)
        return self.expmap

    def get_signal_map(self):
        if self.countrate_map is None:
            cat = self.get_pholist()
            xpmap = self.get_expmap()
            mask = self.get_mask()
            count_map = get_map_from_points(cat, self.nside,
                                            ra_name='raj2000',
                                            dec_name='dej2000')
            self.countrate_map = np.zeros(self.npix)
            goodpix = mask > 0.0
            self.countrate_map[goodpix] = count_map[goodpix] / xpmap[goodpix]
            pixA = hp.nside2pixarea(self.nside)
            self.countrate_map *= 1/pixA
        return [self.countrate_map]

    def get_mask(self):
        if self.mask is None:
            self.mask = np.ones(self.npix)
            xpmap = self.get_expmap()
            self.mask[xpmap <= self.explimit] = 0
            if self.mask_external is not None:
                msk = hp.ud_grade(hp.read_map(self.mask_external),
                                  nside_out=self.nside)
                self.mask[msk <= 0] = 0
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.nl_coupled = np.zeros([1, 3*self.nside])
        return self.nl_coupled

    def get_dtype(self):
        return 'generic'

    def get_spin(self):
        return 0
