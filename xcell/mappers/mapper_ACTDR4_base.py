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
        self._get_defaults(config)
        self.file_map = config['file_map']
        self.file_mask = config['file_mask']
        self.lmax = config.get('lmax', 6000)
        self.signal_map = None
        self.mask = None
        self.pixell_mask = None
        self.noise = None
        self.cross_noise = None
        self.weights = None
        self.beam_info = None

    def get_signal_map(self):
        return NotImplementedError("Do not use base class")

    def get_mask(self):
        return NotImplementedError("Do not use base class")

    def get_noise(self):
        if self.pixell_mask is None:
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
    
    def _beam_gaussian(self, ell, fwhm_amin):
        sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
        return np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)

    def get_beam(self):
        if self.beam is None:
            if self.beam_info is None:  # No beam
                self.beam = np.ones(3*self.nside)
            else:
                ell = np.arange(3*self.nside)
                self.beam = self._beam_gaussian(ell, self.beam_info)
        return self.beam

