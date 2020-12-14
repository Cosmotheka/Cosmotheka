from .mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table
import numpy as np
import healpy as hp
import pymaster as nmt


class MapperDESY1gc(MapperBase):
    def __init__(self, config):
        """
        Data source:
        https://des.ncsa.illinois.edu/releases/y1a1/key-catalogs/key-shape
        config - dict
          {'data_catalogs':'/home/zcapjru/PhD/Data/DES_redm/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits',
           'file_mask':'/home/zcapjru/PhD/Data/DES_redm/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
           'file_nz':'/home/zcapjru/PhD/Data/DES_redm/2pt_NG_mcal_1110.fits',
           'bin':1,
           'nside':nside,
           'mask_name': 'mask_DESgc_1'}
        """

        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.mask_threshold = config.get('mask_threshold', 0.5)
        self.bin_edges = {
            '1': [0.15, 0.30],
            '2': [0.30, 0.45],
            '3': [0.45, 0.60],
            '4': [0.60, 0.75],
            '5': [0.75, 0.90]}

        self.cat_data = Table.read(self.config['data_catalogs']).to_pandas()
        self.nz = fits.open(self.config['file_nz'])[7].data
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.bin = config['bin']
        self.z_edges = self.bin_edges['{}'.format(self.bin)]

        self.cat_data = self._bin_z(self.cat_data)
        self.w_data = self._get_weights(self.cat_data)
        self.nmap_data = self._get_counts_map(self.cat_data, w=self.w_data)
        self.mask = None
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.nmt_field = None

    def _bin_z(self, cat):
        if 'ZREDMAGIC' in cat:
            z_key = 'ZREDMAGIC'
        else:
            z_key = 'Z'

        return cat[(cat[z_key] >= self.z_edges[0]) &
                   (cat[z_key] < self.z_edges[1])]

    def _get_weights(self, cat):
        return np.array(cat['weight'].values)

    def _get_counts_map(self, cat, w=None, nside=None):
        if nside is None:
            nside = self.nside
        npix = hp.nside2npix(nside)
        ipix = hp.ang2pix(nside, cat['RA'], cat['DEC'],
                          lonlat=True)
        numcount = np.bincount(ipix, weights=w, minlength=npix)
        return numcount

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.config['file_mask'], verbose=False)
            self.mask = hp.ud_grade(self.mask, nside_out=self.nside)
            goodpix = self.mask > self.mask_threshold
            self.mask[~goodpix] = 0
        return self.mask

    def get_nz(self, num_z=200):
        if self.dndz is None:
            # Equivalent to getting columns 1 and 3 in previous code
            z = self.nz['Z_MID']
            pz = self.nz['BIN%d' % (self.bin)]
            # Calculate z bias
            dz = 0
            z_dz = z - dz
            # Set to 0 points where z_dz < 0:
            sel = z_dz >= 0
            z_dz = z_dz[sel]
            pz = pz[sel]
        return np.array([z_dz, pz])

    def get_signal_map(self):
        if self.delta_map is None:
            self.mask = self.get_mask()
            self.delta_map = np.zeros(self.npix)
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            goodpix = self.mask > 0
            nm = self.mask*N_mean
            self.delta_map[goodpix] = (self.nmap_data[goodpix])/(nm[goodpix])-1
        return [self.delta_map]

    def get_nmt_field(self):
        if self.nmt_field is None:
            signal = self.get_signal_map()
            mask = self.get_mask()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter=0)
        return self.nmt_field

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.mask = self.get_mask()
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = np.sum(self.w_data**2) / np.sum(self.w_data)
            N_ell = correction * np.sum(self.mask) / self.npix / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled