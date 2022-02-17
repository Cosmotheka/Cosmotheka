from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperMWhiteLRGgc(MapperBase):
    def __init__(self, config):
        """
        Data source:
        https://des.ncsa.illinois.edu/releases/y1a1/key-catalogs/key-shape
        config - dict
          {'file_map': '/mnt/extraspace/gravityls_3/data/mwhite-ForOxford/maps/lrg_s01_del.hpx0256.fits',
           'file_mask': '/mnt/extraspace/gravityls_3/data/mwhite-ForOxford/maps/lrg_s01_msk.hpx0256.fits',
           'file_nz': '/mnt/extraspace/gravityls_3/data/mwhite-ForOxford/data/lrg_s01_dndz.txt'
           'SN': uncoupled noise
           'nside':nside,
           'mask_name': 'mask_MWhiteLRGgc'}
        """

        self._get_defaults(config)
        self.mask_threshold = config.get('mask_threshold', 0.0)
        self.npix = hp.nside2npix(self.nside)
        self.mask = None
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.cls_cov = None
        self.SN = config.get('SN', 0)

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.config['file_mask'])
            self.mask = hp.ud_grade(self.mask, nside_out=self.nside)
            # Cap it
            goodpix = self.mask > self.mask_threshold
            self.mask[~goodpix] = 0
        return self.mask

    def get_nz(self, dz=0):
        if self.dndz is None:
            z, nz = np.loadtxt(self.config['file_nz'], unpack=True)
            self.dndz = {'z_mid': z,
                         'nz': nz}
        return self._get_shifted_nz(dz)

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = hp.read_map(self.config['file_map'])
            self.delta_map = hp.ud_grade(self.delta_map, nside_out=self.nside)
        return [self.delta_map]

    def get_nl_coupled(self):
        # We follow what we do in MapperP18CMBK.
        # SN is the uncoupled noise. To "couple" it, multiply by the mean of
        # the squared mask. This will account for the factor that will be
        # divided for the coviariance.

        if self.nl_coupled is None:
            nl = self.SN * np.mean(self.get_mask()**2)
            self.nl_coupled =  nl * np.ones((1, 3 * self.nside))
        return self.nl_coupled


    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
