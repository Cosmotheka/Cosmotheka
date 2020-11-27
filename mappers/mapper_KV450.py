from mapper_base import MapperBase

from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class MapperKV450(MapperBase):
    def __init__(self, config):
        
        self.config = config
        self.column_names = ['SG_FLAG', 'GAAP_Flag_ugriZYJHKs',
                             'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             'bias_corrected_e1', 'bias_corrected_e2',
                             'weight']
        self.zbin_edges = {
        '1':[0.1, 0.3],
        '2':[0.3, 0.5],
        '3':[0.5, 0.7],
        '4':[0.7, 0.9],
        '5':[0.9, 1.2]}
        
        self.cat_data = []
        for file_data in self.config['data_catalogs']:
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            with fits.open(file_data) as f:
                self.cat_data.append(Table.read(f).to_pandas())
                #cat = pd.DataFrame()
                #cat = Table.read(f), format='fits')
                #cat = cat[self.column_names]
                #self.cat_data.append(cat)
        
        print('Catalogs loaded')
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.zbin = config['zbin']
        self.z_edges = self.zbin_edges['{}'.format(self.zbin)]
        print('z edges ' , self.z_edges)
        self.cat_data = pd.concat(self.cat_data)
        print('data concatenated')
        self.cat_data = self._bin_z(self.cat_data)
        print('Data binned')
        self._remove_additive_bias()
        print('Additive biased removed')
        self._remove_multiplicative_bias() 
        print('Multiplicative bias removed')

        self.dndz = np.loadtxt(self.config['file_nz'], unpack=True)
        self.shear_map  = None
        self.psf_map    = None
        self.star_map   = None
        self.star_mask  = None
        self.galaxy_mask= None
        self.nl_coupled = None
        self.nmt_field  = None

    def _bin_z(self, cat):
        z_key = 'Z_B'   
        return cat[(cat[z_key] >= self.z_edges[0]) &
                   (cat[z_key] < self.z_edges[1])]

    def _remove_additive_bias(self):
        sel_gals = self.cat_data['SG_FLAG'] == 1
        print(np.mean(self.cat_data[sel_gals]['bias_corrected_e1']))
        print(np.mean(self.cat_data[sel_gals]['bias_corrected_e2']))
        self.cat_data[sel_gals]['bias_corrected_e1'] -= np.mean(self.cat_data[sel_gals]['bias_corrected_e1'])
        self.cat_data[sel_gals]['bias_corrected_e2'] -= np.mean(self.cat_data[sel_gals]['bias_corrected_e2'])

    def _remove_multiplicative_bias(self):
        # Values from Table 2 of 1812.06076 (KV450 cosmo paper)
        m = (-0.017, -0.008, -0.015, 0.010, 0.006)
        sel_gals = self.cat_data['SG_FLAG'] == 1
        self.cat_data[sel_gals]['bias_corrected_e1'] /= 1 + m[self.zbin]
        self.cat_data[sel_gals]['bias_corrected_e2'] /= 1 + m[self.zbin]
    
    def _get_galaxy_data(self):
        sel = self.cat_data['SG_FLAG'] == 1
        return self.cat_data[sel]

    def _get_star_data(self):
        sel = self.cat_data['SG_FLAG'] == 0
        return self.cat_data[sel]
    
    def get_shear_map(self, nside=None):
        if self.shear_map is None:
            if nside is None:
                nside = self.nside
            npix = hp.nside2npix(nside)    
            phi = np.radians(self.cat_data['ALPHA_J2000'])
            theta = np.radians(90 - self.cat_data['DELTA_J2000'])
            ipix = hp.ang2pix(nside, theta, phi)

            we1 = np.bincount(ipix, weights=self.cat_data['weight']*self.cat_data['bias_corrected_e1'], minlength=npix)

            we2 = np.bincount(ipix, weights=self.cat_data['weight']*self.cat_data['bias_corrected_e2'], minlength=npix)

            w2s2 = np.bincount(ipix, weights=self.cat_data['weight']**2 * 0.5 * (self.cat_data['bias_corrected_e1']**2 + self.cat_data['bias_corrected_e2']**2), minlength=npix)
            self.shear_map = [we1, we2, w2s2]
        return self.shear_map

    def get_psf_map(self, nside=None):
        if self.psf_map is None:
            if nside is None:
                nside = self.nside
            npix = hp.nside2npix(nside)    
            phi = np.radians(self.cat_data['ALPHA_J2000'])
            theta = np.radians(90 - self.cat_data['DELTA_J2000'])
            ipix = hp.ang2pix(nside, theta, phi)

            we1 = np.bincount(ipix, weights=self.cat_data['weight']*self.cat_data['PSF_e1'], minlength=npix)

            we2 = np.bincount(ipix, weights=self.cat_data['weight']*self.cat_data['PSF_e2'], minlength=npix)

            w2s2 = np.bincount(ipix, weights=self.cat_data['weight']**2 * 0.5 * (self.cat_data['PSF_e1']**2 + self.cat_data['PSF_e2']**2), minlength=self.npix)
            self.psf_map = [we1, we2, w2s2]
        return self.psf_map

    def get_star_map(self, nside=None):
        if self.star_map is None:
            if nside is None:
                nside = self.nside
            npix = hp.nside2npix(nside)    
            phi = np.radians(self.cat_data['ALPHA_J2000'])
            theta = np.radians(90 - self.cat_data['DELTA_J2000'])
            ipix = hp.ang2pix(nside, theta, phi)
            
            self.star_map = np.bincount(ipix, minlength=self.npix)
        return self.star_map

    def get_star_mask(self, nside=None, w2=True):
        if self.star_mask is None:
            if nside is None:
                nside = self.nside
            npix = hp.nside2npix(nside)  
            phi = np.radians(self.cat_data['ALPHA_J2000'])
            theta = np.radians(90 - self.cat_data['DELTA_J2000'])
            ipix = hp.ang2pix(nside, theta, phi)
            
            self.star_mask = [np.bincount(ipix, weights=self.cat_data['weight'], minlength=npix)]
            if w2:
                w2 = np.bincount(ipix, weights=self.cat_data['weight']**2, minlength=npix)
                self.star_mask.append(w2)
        return self.star_mask

    def get_galaxy_mask(self, nside=None):
        if self.galaxy_mask is None:
            if nside is None:
                nside = self.nside
            npix = hp.nside2npix(nside)    
            phi = np.radians(self.cat_data['ALPHA_J2000'])
            theta = np.radians(90 - self.cat_data['DELTA_J2000'])
            ipix = hp.ang2pix(nside, theta, phi)
            
            self.galaxy_mask = np.bincount(ipix, weights=self.cat_data['weight'], minlength=npix)
        return self.galaxy_mask

    def get_nmt_field(self):
        if self.nmt_field is None:
            signal = self.get_signal_map()
            mask = self.get_mask()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = np.sum(self.w_data**2) / np.sum(self.w_data)
            N_ell = correction * np.sum(self.mask) / self.npix / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled
