from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
import os


class MapperP18tSZ(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits', 
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_tSZ',
         'nside':512}
        """
        self._get_defaults(config)
        self.beam = config.get('beam', 0.00291*np.ones(3*self.nside))
        self.file_map = config['file_map']
        self.file_mask = config['file_mask']
        self.mode = 'FULL'
        
        # Defaults
        self.signal_map = None
        self.maps = {'FULL': None,
                    'FIRST': None,
                    'LAST': None, 
                    'DIFF':None}
        self.nl_coupled = None
        self.mask = None
        
    def _get_bands(self):
        while i <= 3 * self.nside:
            ells.append(round(i))
            #i = i*(1+i/(3 * nside))
            i = i+20*(1+i/240)

        self.bands = nmt.NmtBin.from_edges(ells[:-1], ells[1:])
        self.ell_arr = bands.get_effective_ells()
        return 
        
    def get_signal_map(self, mode=None):
        if mode is None:
            mode = self.mode
        if self.signal_map is None:
            signal_maps = fits.open(self.file_map)[1].data
            self.maps['FULL'] = signal_maps['FULL']
            self.maps['FIRST'] = signal_maps['FIRST']
            self.maps['LAST'] = signal_maps['LAST']
            self.maps['DIFF'] = signal_maps['FIRST']-signal_maps['LAST']
            self.signal_map = self.maps[mode]
            self.signal_map = [hp.ud_grade(self.signal_map,
                                           nside_out=self.nside)]
        return self.signal_map 

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.file_mask)
            self.mask = hp.ud_grade(self.mask,
                                    nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.signal_map = self.get_signal_map(mode='DIFF')
            field = self.get_nmt_field()
            self.nl_coupled = nmt.compute_coupled_cell(field, field)/4
            
        return self.nl_coupled

    def get_beam(self):
        return self.beam
    
    def get_ell(self):
        return np.arange(3 * self.nside)

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
