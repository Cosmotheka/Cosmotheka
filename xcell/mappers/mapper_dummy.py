from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import os

class MapperDummy(MapperBase):
    def __init__(self):
        """
        config - dict
          {'data_catalogs': ***,
           'random_catalogs': ***,
           'z_edges': ***,
           'nside': ***,
           'nside_mask': ***,
           'mask_name': ***}
        """
        self.n_points_data = 40000
        self.n_points_random = 50*self.n_points_data
        self.nside =64
        
        self.nmap_d = self._get_random_map(self.n_points_data)
        self.nmap_r = self._get_random_map(self.n_points_random)
        # R/D ratio
        self.alpha = np.sum(self.nmap_d)/np.sum(self.nmap_r)
            
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None
        self.nmt_field = None

    def get_signal_map(self):
        if self.delta_map is None:
            # Non-empty pixels
            goodpix = self.nmap_r > 0
            # Overdensity
            self.delta_map = np.zeros_like(self.nmap_d)
            self.delta_map[goodpix] = self.nmap_d[goodpix]/(self.alpha*self.nmap_r[goodpix]) - 1
        return [self.delta_map]  
    
    def get_mask(self):
        if self.mask is None:
            self.mask = self.alpha*self.nmap_r
        return self.mask 

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            # Coupled analytical noise bias
            a_pix = 4*np.pi/len(self.nmap_d)
            N_ell = (np.sum(self.nmap_d) + self.alpha**2*np.sum(self.nmap_r))
            N_ell *= a_pix**2/(4*np.pi)
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled
    
    def _get_random_map(self, n):
        
        """ Generates and returns a map of resolution `nside`
        with `n` randomly positioned points between declinations
        `dec_min` and `dec_max` and right ascensions `ra_min`
        and `ra_max`. Each pixel in the returned map contains
        the number of points falling inside it.
        """
        dec_min=10.
        dec_max=70.
        ra_min=0.
        ra_max=360.
            
        # Generate cos(theta) and phi
        npix = hp.nside2npix(self.nside)
        cthmin = np.cos(np.radians(90-dec_min))
        cthmax = np.cos(np.radians(90-dec_max))
        cth = cthmin + (cthmax-cthmin)*np.random.rand(n)
        th = np.arccos(cth)
        ph = np.radians(ra_min + (ra_max-ra_min)*np.random.rand(n))
        
        # HEALPix pixel indices
        ipix = hp.ang2pix(self.nside, th, ph)
        # Number counts map
        nmap = np.bincount(ipix, minlength=npix)
        return nmap.astype(float)

    def get_dtype(self):
        return 'galaxy_density'
    
    def get_spin(self):
        return '0'
