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
        if os.path.isfile('KV450_lite_cat_0.pkl'):
            print('loading lite cats')
            for i in range(len(self.config['data_catalogs'])):
                 self.cat_data.append(pd.read_pickle('KV450_lite_cat_{}.pkl'.format(i)))
        else:
            print('loading full cats and making lite versions')
            for i, file_data in enumerate(self.config['data_catalogs']):
                if not os.path.isfile(file_data):
                    raise ValueError(f"File {file_data} not found")
                with fits.open(file_data) as f:
                    table = Table.read(f).to_pandas()[self.column_names] 
                    table.to_pickle('KV450_lite_cat_{}.pkl'.format(i))
                    self.cat_data.append(table)
        
        print('Catalogs loaded')
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.zbin = config['zbin']
        self.z_edges = self.zbin_edges['{}'.format(self.zbin)]
        print('z edges ' , self.z_edges)
        self.cat_data = pd.concat(self.cat_data)
        print('data concatenated')
        self.cat_data = self._get_GAAP_data()
        print('removed GAAP')
        self.cat_data = self._bin_z(self.cat_data)
        print('Length of galaxies: ', len(self._get_galaxy_data()['ALPHA_J2000']))
        print('Length of stars: ', len(self._get_star_data()['ALPHA_J2000']))
        print('Data binned')
        self._remove_additive_bias()
        print('Additive biased removed')
        self._remove_multiplicative_bias() 
        print('Multiplicative bias removed')

        self.dndz = np.loadtxt(self.config['file_nz'], unpack=True)
        self.signal_map  = None
        self.psf_map     = None
        self.shear_map   = None
        self.star_map    = None
        
        self.mask        = None
        self.star_mask   = None 
        self.galaxy_mask = None
        
        self.nmt_field   = None
        self.nl_coupled  = None
        self.shear_nl_coupled = None
        self.psf_nl_coupled   = None 
        self.stars_nl_coupled = None 

    def _bin_z(self, cat):
        z_key = 'Z_B'   
        return cat[(cat[z_key] >= self.z_edges[0]) &
                   (cat[z_key] < self.z_edges[1])]
    
    def _get_counts_map(self, data, w=None, nside=None):
        if nside is None:
            nside = self.nside
            
        phi = np.radians(data['ALPHA_J2000'])
        theta = np.radians(90 - data['DELTA_J2000'])
        ipix = hp.ang2pix(self.nside, theta, phi)
        npix = hp.nside2npix(nside) 
        
        numcount = np.bincount(ipix,  weights=w , minlength=npix )
        
        return numcount

    def _remove_additive_bias(self):
        sel_gals = self.cat_data['SG_FLAG'] == 1
        #print(np.mean(self.cat_data[sel_gals]['bias_corrected_e1']))
        #print(np.mean(self.cat_data[sel_gals]['bias_corrected_e2']))
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
    
    def _get_GAAP_data(self):
        sel = self.cat_data['GAAP_Flag_ugriZYJHKs'] == 0
        return self.cat_data[sel]

    def _get_star_data(self):
        sel = self.cat_data['SG_FLAG'] == 0
        return self.cat_data[sel]
    
    def get_signal_map(self, map_mode = 'shear'):
        if map_mode == 'shear':
            print('Calculating shear spin-2 field')
            self.signal_map = self._get_shear_map()
            return self.signal_map
        elif map_mode == 'PSF': 
            print('Calculating PSF spin-2 field')
            self.signal_map = self._get_psf_map()
            return self.signal_map
        elif map_mode == 'stars': 
            print('Calculating star density spin-0 field')
            self.signal_map = self._get_star_map()
            return self.signal_map
        else:
            print('Unrecognized mode. Please choose between: shear, PSF or stars.')
            return
              
    def _get_shear_map(self):
        if self.shear_map is None:
            data = self._get_galaxy_data()
            we1 = self._get_counts_map(data, w=data['weight']*data['bias_corrected_e1'], nside=None)
            we2 = self._get_counts_map(data, w=data['weight']*data['bias_corrected_e2'], nside=None)
            self.shear_map = [we1, we2]
        return self.shear_map

    def _get_psf_map(self):
        if self.psf_map is None:
            data = self._get_galaxy_data()
            we1 = self._get_counts_map(data, w=data['weight']*data['PSF_e1'], nside=None)
            we2 = self._get_counts_map(data, w=data['weight']*data['PSF_e2'], nside=None)
            self.psf_map = [we1, we2]
        return self.psf_map

    def _get_star_map(self):
        if self.star_map is None:
            data = self._get_star_data()
            self.star_map = [self._get_counts_map(data)]
        return self.star_map

    def get_mask(self, mask_mode='shear'):
        if  mask_mode == 'shear':
            print('Using galaxies mask')
            self.mask = self._get_galaxy_mask()
            return self.mask
        elif mask_mode == 'PSF': 
            print('Using galaxies mask')
            self.mask = self._get_galaxy_mask()
            return self.mask
        elif mask_mode == 'stars': 
            print('Using stars mask')
            self.mask = self._get_star_mask()
            return self.mask
        else:
            print('Unrecognized mode. Please choose between: shear, PSF or stars.')
            return
    
    def _get_star_mask(self):
        if self.star_mask is None:
            data = self._get_star_data()
            self.star_mask = self._get_counts_map(data, w = data['weight'])
        return self.star_mask
    
    def _get_galaxy_mask(self):
        if self.galaxy_mask is None:
            data = self._get_galaxy_data()
            self.galaxy_mask = self._get_counts_map(data, w = data['weight'])
        return self.galaxy_mask

    def get_nmt_field(self, mode = 'shear'):
        signal = self.get_signal_map(map_mode = mode)
        mask = self.get_mask(mask_mode = mode)
        self.nmt_field = nmt.NmtField(mask, signal, n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self, nl_mode = 'shear'):
        if  nl_mode == 'shear':
            print('Calculating shear nl coupled')
            self.nl_coupled = self._get_shear_nl_coupled()
            return self.nl_coupled
        elif nl_mode == 'PSF': 
            print('Calculating psf nl coupled')
            self.nl_coupled = self._get_psf_nl_coupled()
            return self.nl_coupled
        elif nl_mode == 'stars': 
            print('Calculating stars nl coupled')
            self.nl_coupled = self._get_star_nl_coupled()
            return self.nl_coupled
        else:
            print('Unrecognized mode. Please choose between: shear, PSF or stars.')
            return
    
    def _get_shear_nl_coupled(self, mode = 'shear'):
        if self.shear_nl_coupled is None:
            data = self._get_galaxy_data()
            w2s2 = self._get_counts_map(data, w=data['weight']**2 * 0.5 * (data['bias_corrected_e1']**2 + data['bias_corrected_e2']**2), nside=None)
            N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix 
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # Ylm = for l < spin
            self.shear_nl_coupled = np.array([nl, 0 * nl, 0 * nl, nl])

        return self.shear_nl_coupled
    
    def _get_psf_nl_coupled(self):
        if self.psf_nl_coupled is None:
            data = self._get_galaxy_data()
            w2s2 = self._get_counts_map(data, w = data['weight']**2 * 0.5 * (data['PSF_e1']**2 + data['PSF_e2']**2))
            N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # Ylm = for l < spin
            self.psf_nl_coupled = np.array([nl, 0 * nl, 0 * nl, nl])

        return self.psf_nl_coupled
    
    def _get_stars_nl_coupled(self):
        if self.stars_nl_coupled is None:
            data = self._get_star_data()
            N_mean = np.sum(self.star_map)/np.sum(self.star_mask)
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = np.sum(data['weight']**2) / np.sum(data['weight'])
            N_ell = correction * np.sum(self.star_mask) / self.npix / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))

        return self.stars_nl_coupled
