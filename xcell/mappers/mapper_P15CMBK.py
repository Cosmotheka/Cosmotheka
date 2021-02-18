from .mapper_base import MapperBase
import pandas as pd
import numpy as np
import healpy as hp


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

        # Read noise file
        self.noise = pd.read_table(self.config['file_noise'],
                                   names=['l', 'Nl', 'Nl+Cl'],
                                   sep=" ", encoding='utf-8')
        # Galactic-to-celestial coordinate rotator
        self.r = hp.Rotator(coord=['G', 'C'])

        # Defaults
        self.signal_map = None
        self.nl_coupled = None
        self.mask = None
        self.cl_fid = None

    def get_signal_map(self):
        if self.signal_map is None:
            # Read alms
            self.klm = hp.read_alm(self.config['file_klm'])
            self.klm = self.r.rotate_alm(self.klm)
            self.signal_map = hp.alm2map(self.klm, self.nside,
                                         verbose=False)
        return [self.signal_map]

    def get_mask(self):
        if self.mask is None:
            self.mask = hp.read_map(self.config['file_mask'],
                                    verbose=False)
            self.mask = self.r.rotate_map_pixel(self.mask)
            # Binerize
            self.mask[self.mask < 0.5] = 0
            self.mask[self.mask >= 0.5] = 1.
            #
            self.mask = hp.ud_grade(self.mask,
                                    nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
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

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
