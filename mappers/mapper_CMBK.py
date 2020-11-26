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
        self.mask_name = self.config['mask_name']

        self.klm = []
        self.mask = []
        self.noise = []
  
        self.klm = hp.read_alm(self.config['file_klm'])        
        self.mask = hp.read_map(self.config['file_mask'])  
        self.noise = pd.read_table(self.config['file_noise'], 
                                   names=['l','Nl','Nl+Cl'], sep=" ", encoding='utf-8')
            
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
        return [self.k_map]

    def get_mask(self):
        if self.mask_map is None:
            self.mask_map = self.r.rotate_map_pixel(self.mask)
        return self.mask_map

    def get_nmt_field(self):
        if self.nmt_field is None:
            mask = self.get_mask()
            signal = self.get_signal_map()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter = 0)
        return self.nmt_field

    def get_nl(self):
        if self.nl_coupled is None:
            self.nl_coupled = self.noise['Nl'].values
        return np.array([self.nl_coupled])
    
    def get_cl_fiducial(self):
        if self.cl_fid is None:
            self.cl_fid = self.noise['Nl+Cl'].values - self.noise['Nl'].values
        return self.cl_fid
    
    def get_ells(self):   
        return self.noise['l'].values

