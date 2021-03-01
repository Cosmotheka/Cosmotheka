from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MapperDECaLS(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':['data_Legacy_Survey_BASS-MZLS_galaxies-selection.fits'],
           'mask': 'data_Legacy_footprint_final_mask.fits',
           'comp_map': 'data_Legacy_footprint_completeness_mask_128.fits',
           'stars': 'data_allwise_total_rot_1024.fits',
           'zbin': 0,
           'nside': 128,
           'nside_mask': nside_mask,
           'mask_name': 'mask_DECaLS_SGC'}
        """
        self._get_defaults(config)
        self.nside_fid = 1024

        self.cat_data = []
        self.z_arr_dim = config.get('z_arr_dim', 50)
        self.npix = hp.nside2npix(self.nside)

        for file_data in self.config['data_catalogs']:
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            with fits.open(file_data) as f:
                self.cat_data.append(Table.read(f))
                
        self.cat_data = vstack(self.cat_data)
        
        bin_edges = [[0.00, 0.30],
                     [0.30, 0.45],
                     [0.45, 0.60],
                     [0.60, 0.80]]
        
        self.zbin = config['zbin']
        self.z_edges = bin_edges[self.zbin]
        self.cat_data = self._bin_z(self.cat_data)

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None
        self.comp_map = None
        self.stars = None

    def _bin_z(self, cat):
        return cat[(cat['PHOTOZ_3DINFER'] >= self.z_edges[0]) &
                   (cat['PHOTOZ_3DINFER'] < self.z_edges[1])]

    def get_nz(self, convolve=True, dz=0):
        #if self.dndz is None:
        h, b = np.histogram(self.cat_data['PHOTOZ_3DINFER'], bins=self.z_arr_dim)
        z_arr = 0.5 * (b[:-1] + b[1:])
        self.dndz = np.array([z_arr, h])
        lorentzian = self._get_lorentzian(z_arr)
        print(lorentzian)
        conv = np.sum(self.dndz[1]*lorentzian, axis=1)
        # Normalization
        conv /= np.sum(conv) 
        if convolve:
            self.dndz = np.array([self.dndz[0], conv])
        else:
            self.dndz = np.array([self.dndz[0],
                                  self.dndz[1]/np.sum(self.dndz[1])])
        #####
            
        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])

    def _get_lorentzian(self, z_arr):
        # a m s 
        params = np.array([[1.257, -0.0010, 0.0122],
                           [1.104, 0.0076, 0.0151],
                           [1.476, -0.0024, 0.0155],
                           [2.019, -0.0042, 0.0265]])
        [a, m, s] =  params[self.zbin]
        lorentzian = np.zeros([len(z_arr), len(z_arr)])
        # NOTE: sample more points
        for zs_i, zs in enumerate(z_arr):
            for zp_i, zp in enumerate(z_arr):
                lorentzian[zp_i, zs_i] = ((zp-zs-m)/(s**2))**2
                lorentzian[zp_i, zs_i] /= 2*a
                lorentzian[zp_i, zs_i] += 1
                lorentzian[zp_i, zs_i] = lorentzian[zp_i, zs_i]**(-a)                  
        return lorentzian               
    
    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            self.comp_map = self._get_comp_map() 
            self.stars = self._get_stars()
            # PROBLEM: correction factor too big for threshold
            nmap_data = get_map_from_points(self.cat_data, self.nside)
            mean_n = self._get_mean_n(nmap_data)
            self.mask = self.get_mask()
            goodpix = self.mask > 0
            self.delta_map[goodpix] = nmap_data[goodpix]/self.comp_map[goodpix]
            self.delta_map[goodpix] /= mean_n
            self.delta_map[goodpix] += -1
        return [self.delta_map]

    def _get_mean_n(self, nmap):
        goodpix = self.comp_map > 0.95 
        goodpix *= self.stars < 8520
        goodpix *= self.mask > 0
        n_mean = np.mean(nmap[goodpix])
        return n_mean
    
    def _get_stars(self):
        if self.stars is None:
            self.stars = hp.read_map(self.config['stars'])
            area_ratio = (self.nside_fid/self.nside)**2 
            self.stars = area_ratio * hp.ud_grade(self.stars,
                                                 nside_out=self.nside)
            # correct for the fact that threshold is given 
            # in stars per degree^2 as opposed to pixel
            self.stars *= (360**2/np.pi)/(12*self.nside**2)
        return self.stars
    
    def _get_comp_map(self):
        if self.comp_map is None:
            self.comp_map = hp.read_map(self.config['comp_map'])
            area_ratio = 1# (self.nside_fid/self.nside)**2
            self.comp_map = area_ratio * hp.ud_grade(self.comp_map,
                                                 nside_out=self.nside)
        return self.comp_map
    
    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.config['mask'])
            area_ratio = (self.nside_fid/self.nside)**2
            self.mask = area_ratio * hp.ud_grade(self.mask,
                                                 nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if self.nside < 4096:
                print('calculing nl from weights')
                pixel_A = 4*np.pi/self.npix
                mask = self.get_mask()
                w2_data = get_map_from_points(self.cat_data, self.nside,
                                              w=self.w_data**2)
                w2_random = get_map_from_points(self.cat_random, self.nside,
                                                w=self.w_random**2)
                goodpix = mask > 0
                N_ell = (w2_data[goodpix].sum() +
                         self.alpha**2*w2_random[goodpix].sum())
                N_ell *= pixel_A**2/(4*np.pi)
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
            else:
                print('calculating nl from mean cl values')
                f = self.get_nmt_field()
                cl = nmt.compute_coupled_cell(f, f)[0]
                N_ell = np.mean(cl[2000:2*self.nside])
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
