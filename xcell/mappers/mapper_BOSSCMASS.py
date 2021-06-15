from .mapper_base import MapperBase
from .mapper_base import MapperSDSS
from astropy.io import fits
from astropy.table import Table, vstack
from .utils import get_map_from_points
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class MapperBOSSCMASS(MapperSDSS):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':[data_path+'BOSSCMASS/galaxy_DR12v5_CMASS_North.fits.gz'], 
          'random_catalogs':[data_path+'BOSSCMASS/random1_DR12v5_CMASS_North.fits.gz'],
          'z_edges':[0, 1.5],
          'nside':nside, 'nside_mask': nside_mask, 'mask_name': 'mask_CMASS_NGC_1'}
        """
        self._get_defaults(config)

        self.cats = {'data': None, 'random': None}

        self.z_arr_dim = config.get('z_arr_dim', 50)
        self.nside_mask = config.get('nside_mask', 512)
        self.npix = hp.nside2npix(self.nside)
        self.mask_path = config['mask_path']
        self.z_edges = config['z_edges']

        self.ws = {'data': None, 'random': None}
        self.alpha = None

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None

    def _bin_z(self, cat):
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]
        
    def get_nz(self, dz=0):
        if self.dndz is None:
            cat_data = self.get_catalog(mod='data')
            w_data = self._get_w(mod='data')
            h, b = np.histogram(cat_data['Z'], bins=self.z_arr_dim,
                                weights=w_data, range=[0.5, 2.5])
            self.dndz = np.array([0.5 * (b[:-1] + b[1:]), h])
        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])
    
    def _get_w(self, mod='data'):
        #Could make this more general and pass it to the superclass
        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            if mod == 'data':
                w = np.array(cat['WEIGHT_SYSTOT'])
                w *= np.array(cat['WEIGHT_CP'])
                w *= np.array(cat['WEIGHT_NOZ'])
            elif mod == 'random':
                w = np.ones_like(cat['RA'])
            self.ws[mod] = w # FKP left out
        return self.ws[mod]
    
    def get_mask(self):
        if self.mask is None:
            cat_random = self.get_catalog(mod='random')
            w_random = self._get_w(mod='random')
            alpha = self._get_alpha()
            self.mask = get_map_from_points(cat_random,
                                            self.nside_mask,
                                            w=w_random)
            self.mask *= alpha
            # Account for different pixel areas
            area_ratio = (self.nside_mask/self.nside)**2
            self.mask = area_ratio * hp.ud_grade(self.mask,
                                                 nside_out=self.nside)
        return self.mask
    #def get_mask(self):
    #    if self.mask is None:
    #        self.mask = hp.read_map(self.mask_path, verbose=False)
    #        area_ratio = (self.nside_mask/self.nside)**2
    #        self.mask = area_ratio * hp.ud_grade(self.mask,
    #                                             nside_out=self.nside)
    #    return self.mask

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
