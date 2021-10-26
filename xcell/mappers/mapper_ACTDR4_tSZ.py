from .mapper_ACTDR4_base import MapperACTDR4Base
from scipy.interpolate import interp1d
from pixell import enmap, enplot, reproject, utils, curvedsky 
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MapperACTDR4tSZ(MapperACTDR4Base):
    def __init__(self, config):
        """
        config - dict
        {'file_map':'act_planck_dr4.01_s14s15_D56_lensing_kappa_baseline.fits',
         'file_mask':'act_dr4.01_s14s15_D56_lensing_mask.fits',
         'mask_name': 'mask_ACTK',
         'nside': 1024,
         'lmax': 6000}
        """
        self._get_ACTDR4_defaults(config)
        
    def get_signal_map(self):
        if self.signal_map is None:
            self.signal_map = enmap.read_map(self.file_map)
            self.signal_map = [reproject.healpix_from_enmap(self.signal_map,
                                                           lmax = self.lmax,
                                                           nside = self.nside)]
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
            self.mask = reproject.healpix_from_enmap(self.pixell_mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask

    def get_dtype(self):
        return 'cmb_tSZ'

    def get_spin(self):
        return 0
