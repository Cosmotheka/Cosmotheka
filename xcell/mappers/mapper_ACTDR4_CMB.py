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
        self.file_scmap = config.get('file_scmap', None)
        
    def get_signal_map(self):
        if self.signal_map is None:
            # read source-free map from disk and preprocess (i.e. k-space filter and pixwin)
            self.signal_map = enmap.read_map(self.file_map)[0]
            # Get mask dimensions
            if self.mask_shape is None or self.mask_wcs is None:
                self.get_mask()
            self.signal_map = enmap.extract(self.signal_map,
                                            self.mask_shape,
                                            self.mask_wcs)
            # what does this do exactly?
            self.signal_map = nw.preprocess_fourier(self.signal_map,
                                                    self.mask_shape,
                                                    self.mask_wcs)

            # add in the sources
            if self.file_scmap is not None:
                self.source_map = enmap.read_map(self.file_scmap)[0]
                self.source_map = enmap.extract(self.source_map,
                                                self.mask_shape,
                                                self.mask_wcs)
                self.signal_map += self.source_map
            
            self.pixell_signal_map = self.signal_map
            self.signal_map = reproject.healpix_from_enmap(self.signal_map,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
            # Save dimensions of footprint
            self.mask_shape, self.mask_wcs = footprint.shape, footprint.wcs

            # read in the point source mask, make sure it has the correct shape, and apodize
            if self.file_psmask is not None:
                self.pixell_psmask = enmap.extract(enmap.read_map(self.file_psmask), 
                                                   self.mask_shape, self.mask_wcs)
                self.pixell_psmask = nw.apod_C2(self.pixell_psmask, 18./60.)
                self.pixell_mask *= self.pixell_psmask

            # read in the coadd inverse variance map and make sure it has the correct shape
            if self.file_ivar is not None:
                self.pixell_ivar = enmap.extract(enmap.read_map(self.file_ivar),
                                                 self.mask_shape, self.mask_wcs)
                self.pixell_mask *= self.pixell_ivar
                
            self.mask = reproject.healpix_from_enmap(self.pixell_mask,
                                                     lmax = self.lmax,
                                                     nside = self.nside)
        return self.mask

    def get_dtype(self):
        return 'cmb_temperature'

    def get_spin(self):
        return 0
