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
        self.file_psmask = config.get('file_psmask', None)
        self.file_ivar = config.get('file_ivar', None)
        
    def get_signal_map(self):
        if self.signal_map is None:
            for mapp in self.
    
    # read source-free map from disk and preprocess (i.e. k-space filter and pixwin)
    maps = enmap.read_map(path + f"{mapname_head}{i}_map_srcfree.fits")[0]
    maps = enmap.extract(maps,shape,wcs)
    map_I = nw.preprocess_fourier(maps, shape, wcs)
    del maps
    
    # add in the sources
    source_map = enmap.read_map(path + f"{mapname_head}{i}_srcs.fits")[0]
    source_map = enmap.extract(source_map,shape,wcs)
    map_I = map_I + source_map
    del source_map
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            footprint = enmap.read_map(self.file_mask)
            shape,wcs = footprint.shape, footprint.wcs

            # read in the point source mask, make sure it has the correct shape, and apodize
            psmask = enmap.extract(enmap.read_map(self.file_psmask), shape, wcs)
            psmask = nw.apod_C2(psmask, 18./60.)

            # read in the coadd inverse variance map and make sure it has the correct shape
            ivar = enmap.extract(enmap.read_map(self.file_ivar), shape, wcs)

            self.pixell_mask = footprint*psmask*ivar
            self.mask = reproject.healpix_from_enmap(self.pixell_mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask

    def get_dtype(self):
        return 'cmb_temperature'

    def get_spin(self):
        return 0
