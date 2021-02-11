from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperDESY1gc(MapperBase):
    def __init__(self, config):
        """
        Data source:
        https://des.ncsa.illinois.edu/releases/y1a1/key-catalogs/key-shape
        config - dict
          {'data_catalogs':'/home/zcapjru/PhD/Data/DES_redm/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits',
           'file_mask':'/home/zcapjru/PhD/Data/DES_redm/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
           'file_nz':'/home/zcapjru/PhD/Data/DES_redm/2pt_NG_mcal_1110.fits',
           'zbin':1,
           'nside':nside,
           'mask_name': 'mask_DESgc_1'}
        """

        self._get_defaults(config)
        self.mask_threshold = config.get('mask_threshold', 0.5)
        bin_edges = [[0.15, 0.30],
                     [0.30, 0.45],
                     [0.45, 0.60],
                     [0.60, 0.75],
                     [0.75, 0.90]]
        self.cat_data = Table.read(self.config['data_catalogs'])
        self.nz = fits.open(self.config['file_nz'])[7].data
        self.npix = hp.nside2npix(self.nside)
        self.zbin = config['zbin']
        self.z_edges = bin_edges[self.zbin]
        self.cat_data = self._bin_z(self.cat_data)
        self.w_data = self._get_weights(self.cat_data)
        self.nmap_w = get_map_from_points(self.cat_data, self.nside,
                                          w=self.w_data)
        self.nmap_w2 = get_map_from_points(self.cat_data, self.nside,
                                           w=self.w_data**2)
        self.mask = None
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None

    def _bin_z(self, cat):
        z_key = 'ZREDMAGIC'
        return cat[(cat[z_key] >= self.z_edges[0]) &
                   (cat[z_key] < self.z_edges[1])]

    def _get_weights(self, cat):
        return np.array(cat['weight'])

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.config['file_mask'], verbose=False)
            self.mask = hp.ud_grade(self.mask, nside_out=self.nside)
            # Cap it
            goodpix = self.mask > self.mask_threshold
            self.mask[~goodpix] = 0
        return self.mask

    def get_nz(self, dz=0):
        z = self.nz['Z_MID']
        nz = self.nz['BIN%d' % (self.zbin+1)]
        # Shift distribution by dz and remove z + dz < 0
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])

    def get_signal_map(self):
        if self.delta_map is None:
            self.mask = self.get_mask()
            self.delta_map = np.zeros(self.npix)
            N_mean = np.sum(self.nmap_w)/np.sum(self.mask)
            goodpix = self.mask > 0
            nm = self.mask*N_mean
            self.delta_map[goodpix] = (self.nmap_w[goodpix])/(nm[goodpix])-1
        return [self.delta_map]

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.mask = self.get_mask()
            goodpix = self.mask > 0  # Already capped at mask_threshold
            N_mean = np.sum(self.nmap_w[goodpix])/np.sum(self.mask[goodpix])
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = self.nmap_w2[goodpix].sum()/self.nmap_w[goodpix].sum()
            N_ell = correction * np.mean(self.mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
