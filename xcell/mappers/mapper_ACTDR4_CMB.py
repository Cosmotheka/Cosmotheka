from .mapper_ACTDR4_base import MapperACTDR4Base
from scipy.interpolate import interp1d
from pixell import enmap, enplot, reproject, utils, curvedsky 
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MapperACTDR4CMB(MapperACTDR4Base):
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

    def get_dtype(self):
        return 'cmb_temperature'

    def get_spin(self):
        return 0
