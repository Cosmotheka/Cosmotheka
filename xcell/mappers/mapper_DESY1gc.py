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
        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)
        self.zbin = config['zbin']
        self.z_edges = bin_edges[self.zbin]
        self.w = None
        self.mask = None
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None

    def get_catalog(self):
        if self.cat_data is None:
            self.cat_data = Table.read(self.config['data_catalogs'])
            self.cat_data = self._bin_z(self.cat_data)
        return self.cat_data

    def _bin_z(self, cat):
        z_key = 'ZREDMAGIC'
        return cat[(cat[z_key] >= self.z_edges[0]) &
                   (cat[z_key] < self.z_edges[1])]

    def _get_w(self):
        if self.w is None:
            cat_data = self.get_catalog()
            self.w = np.array(cat_data['weight'])
        return self.w

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.config['file_mask'], verbose=False)
            self.mask = hp.ud_grade(self.mask, nside_out=self.nside)
            # Cap it
            goodpix = self.mask > self.mask_threshold
            self.mask[~goodpix] = 0
        return self.mask

    def get_nz(self, dz=0):
        if self.dndz is None:
            f = fits.open(self.config['file_nz'])[7].data
            z = f['Z_MID']
            nz = f['BIN%d' % (self.zbin+1)]
            self.dndz = np.array([z, nz])
        # Shift distribution by dz and remove z + dz < 0
        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])

    def get_signal_map(self):
        if self.delta_map is None:
            self.mask = self.get_mask()
            cat_data = self.get_catalog()
            w = self._get_w()
            nmap_w = get_map_from_points(cat_data, self.nside,
                                         w=w)
            self.delta_map = np.zeros(self.npix)
            N_mean = np.sum(nmap_w)/np.sum(self.mask)
            goodpix = self.mask > 0
            nm = self.mask*N_mean
            self.delta_map[goodpix] = (nmap_w[goodpix])/(nm[goodpix])-1
        return [self.delta_map]

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat_data = self.get_catalog()
            w = self._get_w()
            nmap_w = get_map_from_points(cat_data, self.nside,
                                         w=w)
            nmap_w2 = get_map_from_points(cat_data, self.nside,
                                          w=w**2)
            self.mask = self.get_mask()
            goodpix = self.mask > 0  # Already capped at mask_threshold
            N_mean = np.sum(nmap_w[goodpix])/np.sum(self.mask[goodpix])
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = nmap_w2[goodpix].sum()/nmap_w[goodpix].sum()
            N_ell = correction * np.mean(self.mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
