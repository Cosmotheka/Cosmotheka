from .mapper_base import MapperBase
import pandas as pd
import numpy as np
import healpy as hp
import pymaster as nmt


class MapperP15CMBK(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {'file_klm':'COM_Lensing_2048_R2.00/dat_klm.fits',
         'file_mask':'COM_Lensing_2048_R2.00/mask.fits.gz',
         'file_noise':'COM_Lensing_2048_R2.00/nlkk.dat',
         'mask_name': 'mask_CMBK',
         'nside':2048}
        """
        self._get_defaults(config)

        # Read alms
        self.klm = hp.read_alm(self.config['file_klm'])
        # Read mask
        self.mask = hp.read_map(self.config['file_mask'],
                                verbose=False)
        # Read noise file
        self.noise = pd.read_table(self.config['file_noise'],
                                   names=['l', 'Nl', 'Nl+Cl'],
                                   sep=" ", encoding='utf-8')
        # Galactic-to-celestial coordinate rotator
        self.r = hp.Rotator(coord=['G', 'C'])

        # Defaults
        self.k_map = None
        self.nl_coupled = None
        self.mask_map = None
        self.nmt_field = None
        self.cl_fid = None

    def get_signal_map(self):
        if self.k_map is None:
            self.k_map = self.r.rotate_alm(self.klm)
            self.k_map = hp.alm2map(self.k_map, self.nside,
                                    verbose=False)
        return [self.k_map]

    def get_mask(self):
        if self.mask_map is None:
            self.mask_map = self.r.rotate_map_pixel(self.mask)
            self.mask_map = hp.ud_grade(self.mask_map,
                                        nside_out=self.nside)
        return self.mask_map

    def get_nmt_field(self):
        if self.nmt_field is None:
            mask = self.get_mask()
            signal = self.get_signal_map()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter=0)
        return self.nmt_field

    def get_nl(self):
        if self.nl_coupled is None:
            self.nl_coupled = self.noise['Nl'].values
        return np.array([self.nl_coupled])

    def get_cl_fiducial(self):
        if self.cl_fid is None:
            self.cl_fid = (self.noise['Nl+Cl'].values -
                           self.noise['Nl'].values)
        return np.array([self.cl_fid])

    def get_ells(self):
        return self.noise['l'].values
