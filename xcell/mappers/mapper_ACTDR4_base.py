from .mapper_base import MapperBase
from scipy.interpolate import interp1d
from pixell import enmap, enplot, reproject, utils, curvedsky 
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MapperACTDR4Base(MapperBase):
    def __init__(self, config):
        self._get_ACTDR4_defaults(config)
        
    def _get_ACTDR4_defaults(self, config):
        self.signal_map = None
        self.mask = None
        self.noise = None
        self.cross_noise = None
        self.weights = None

    def get_signal_map(self):
        if self.signal_map is None:
            self.signal_map = enmap.read_map(self.file_map)
            self.signal_map = reproject.healpix_from_enmap(self.signal_map,
                                                           lmax = self.lmax,
                                                           nside = self.nside)
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            self.mask = enmap.read_map(self.file_mask)
            self.mask = reproject.healpix_from_enmap(self.mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask

    def get_noise(self):
        if self.mask is None:
            self.mask = enmap.read_map(self.file_mask)
            self.mask = reproject.healpix_from_enmap(self.mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask
    
    def get_cross_noise(self):
        if self.mask is None:
            self.mask = enmap.read_map(self.file_mask)
            self.mask = reproject.healpix_from_enmap(self.mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask
    
    def get_weights(self):
        if self.mask is None:
            self.mask = enmap.read_map(self.file_mask)
            self.mask = reproject.healpix_from_enmap(self.mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask

