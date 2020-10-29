from .mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np


class MappereBOSSQSO(MapperBase):
    def __init__(self, config):
        self.config = config

        self.cat_data = []
        self.cat_random = []
        for file_data, file_random in zip(self.config['data_catalogs'],
                                          self.config['random_catalogs']):
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            with fits.open(file_data) as f:
                self.cat_data.append(Table.read(f).to_pandas())
            if not os.path.isfile(file_random):
                raise ValueError(f"File {file_random} not found")
            with fits.open(file_random) as f:
                self.cat_random.append(Table.read(f).to_pandas())
            
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.cat_data = pd.concat(self.cat_data)
        self.cat_random = pd.concat(self.cat_random)

        self.z_edges = config['z_edges']

        self.cat_data = self._bin_z(self.cat_data)
        self.cat_random = self._bin_z(self.cat_random)
        self.w_data = self._get_weights(self.cat_data)
        self.w_random = self._get_weights(self.cat_random)
        self.alpha = np.sum(self.w_data)/np.sum(self.w_random)

        self.dndz = None
        self.delta_map = None

    def _bin_z(self, cat):
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]

    
    def _get_weights(self, cat):
        #cat_FKP = np.array(cat['WEIGHT_FKP'].values) 
        cat_SYSTOT = np.array(cat['WEIGHT_SYSTOT'].values) 
        cat_CP = np.array(cat['WEIGHT_CP'].values) 
        cat_NOZ = np.array(cat['WEIGHT_NOZ'].values)
        weights = cat_SYSTOT*cat_CP*cat_NOZ #FKP left out
        return weights

    def _get_counts_map(self, cat, w):
        ipix = hp.ang2pix(self.nside, cat['RA'], cat['DEC'],
                          lonlat=True)
        numcount = np.bincount(ipix, w, self.npix)
        return numcount
    
    def get_nz(self, num_z=200):
        if self.dndz is None:
            #inputs: cat --> unbinned data
            h, b = np.histogram(self.cat_data['Z'], bins=num_z,
                                weights=self.w_data)
            self.dndz = [h, b[:-1], b[1:]]
        return self.dndz

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            nmap_data = self._get_counts_map(self.cat_data, self.w_data)
            nmap_random = self._get_counts_map(self.cat_random, self.w_random)
            goodpix = nmap_random > 0
            self.delta_map[goodpix] = nmap_data[goodpix] / (self.alpha * nmap_random[goodpix]) - 1
        return self.delta_map

    def get_mask(self):
        pass

    def get_nmt_field(self):
        pass

    def get_nl_coupled(self):
        pass
