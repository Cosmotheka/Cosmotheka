from .utils import get_map_from_points, rotate_mask
from .mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperDESY1gc(MapperBase):
    map_name = 'DESY1gc'

    def __init__(self, config):
        """
        Data source:
        https://des.ncsa.illinois.edu/releases/y1a1/key-catalogs/key-shape
        config - dict
          {'data_catalog':'/home/zcapjru/PhD/Data/DES_redm/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits',
           'file_mask':'/home/zcapjru/PhD/Data/DES_redm/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
           'file_nz':'/home/zcapjru/PhD/Data/DES_redm/2pt_NG_mcal_1110.fits',
           'zbin':1,
           'nside':nside,
           'mask_name': 'mask_DESgc_1'}
        """

        self._get_defaults(config)
        self.rot = self._get_rotator('C')
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
        self.dndz = None
        self.nl_coupled = None

    def get_catalog(self):
        if self.cat_data is None:
            self.cat_data = Table.read(self.config['data_catalog'])
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

    def _get_mask(self):
        mask = hp.read_map(self.config['file_mask'])
        mask[mask == hp.UNSEEN] = 0
        mask = rotate_mask(mask, self.rot)
        mask = hp.ud_grade(mask, nside_out=self.nside)
        # Cap it
        goodpix = mask > self.mask_threshold
        mask[~goodpix] = 0
        return mask

    def get_nz(self, dz=0):
        if self.dndz is None:
            f = fits.open(self.config['file_nz'])[7].data
            self.dndz = {'z_mid': f['Z_MID'],
                         'nz': f['BIN%d' % (self.zbin+1)]}
        return self._get_shifted_nz(dz)

    def _get_signal_map(self):
        mask = self.get_mask()
        cat_data = self.get_catalog()
        w = self._get_w()
        nmap_w = get_map_from_points(cat_data, self.nside,
                                     w=w, rot=self.rot)
        signal_map = np.zeros(self.npix)
        goodpix = mask > 0
        N_mean = np.sum(nmap_w[goodpix])/np.sum(mask[goodpix])
        nm = mask*N_mean
        signal_map[goodpix] = (nmap_w[goodpix])/(nm[goodpix])-1
        return np.array([signal_map])

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat_data = self.get_catalog()
            w = self._get_w()
            nmap_w = get_map_from_points(cat_data, self.nside,
                                         w=w, rot=self.rot)
            nmap_w2 = get_map_from_points(cat_data, self.nside,
                                          w=w**2, rot=self.rot)
            mask = self.get_mask()
            goodpix = mask > 0  # Already capped at mask_threshold
            N_mean = np.sum(nmap_w[goodpix])/np.sum(mask[goodpix])
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = nmap_w2[goodpix].sum()/nmap_w[goodpix].sum()
            N_ell = correction * np.mean(mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
