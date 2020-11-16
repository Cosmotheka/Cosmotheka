from mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class MapperCMBK(MapperBase):
    def __init__(self, config):
        self.config = config

        self.klm = []
        self.mask = []
        self.noise = []
        
        for file_klm, file_mask, file_noise in zip(self.config['data_klm'],
                self.config['data_mask'], self.config['data_noise']):
            
            if not os.path.isfile(file_klm):
                raise ValueError(f"File {file_klm} not found")
            self.klm = hp.read_alm(file_klm)
                
            if not os.path.isfile(file_mask):
                raise ValueError(f"File {file_mask} not found")
            self.mask = hp.read_map(file_mask)
                
            if not os.path.isfile(file_noise):
                raise ValueError(f"File {file_noise} not found")
            self.noise = pd.read_table(file_noise, names=['l','Nl','Nl+Cl'], sep=" ", encoding='utf-8')
            
        self.nside = config['nside']
        self.r = hp.Rotator(coord=['G','C'])
        
        self.k_map      = None
        self.nl_coupled = None
        self.mask_map   = None
        self.nmt_field  = None
        self.cl_fid     = None


    def get_signal_map(self):
        if self.k_map is None:
            self.k_map  = self.r.rotate_alm(self.klm)
            self.k_map =  hp.alm2map(self.k_map, self.nside)
        return self.k_map

    def get_mask(self):
        if self.mask_map is None:
            self.mask_map = self.r.rotate_map_pixel(self.mask)
        return self.mask_map

    def get_nmt_field(self, signal, mask):
        if self.nmt_field is None:
            self.nmt_field = nmt.NmtField(mask, [signal], n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.nl_coupled = self.noise['Nl'].values
            l = self.noise['l'].values
        return l, np.array([self.nl_coupled])
    
    def get_cl_fiducial(self):
        if self.cl_fid is None:
            self.cl_fid = self.noise['Nl+Cl'].values - self.noise['Nl'].values
            l = self.noise['l'].values
        return l, self.cl_fid
