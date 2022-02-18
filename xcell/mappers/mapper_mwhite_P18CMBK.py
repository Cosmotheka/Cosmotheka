from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt


class MapperMWhiteP18CMBK(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {
         'file_map': '/mnt/extraspace/gravityls_3/data/mwhite-ForOxford/maps/P18_lens_kap_filt.hpx2048.fits',
         'file_mask': '/mnt/extraspace/gravityls_3/data/mwhite-ForOxford/maps/P18_lens_msk.hpx2048.fits',
         'file_noise': '/mnt/extraspace/gravityls_3/data/mwhite-ForOxford/maps/P18_lens_nlkk_filt.txt',
         'mask_name': 'mask_MWhiteP18CMBK',
         'nside':4096}
        """
        self._get_defaults(config)
        self.noise = None

        # Defaults
        self.signal_map = None
        self.nl_coupled = None
        self.mask = None
        self.cl_fid = None

    def get_signal_map(self):
        if self.signal_map is None:
            m = hp.read_map(self.config['file_map'])
            self.signal_map = hp.ud_grade(m, nside_out=self.nside)
        return [self.signal_map]

    def get_mask(self):
        if self.mask is None:
            m = hp.read_map(self.config['file_mask'])
            self.mask = hp.ud_grade(m, nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            ell = self.get_ell()
            noise = self._get_noise()
            nl = noise[1]
            nl = interp1d(noise[0], nl, bounds_error=False,
                          fill_value=(nl[0], nl[-1]))(ell)

            # The Nl in the file is decoupled. To "couple" it, multiply by the
            # mean of the squared mask. This will account for the factor that
            # will be divided for the coviariance.
            nl *= np.mean(self.get_mask()**2.)
            self.nl_coupled = np.array([nl])
        return self.nl_coupled

    def _get_noise(self):
        if self.noise is None:
            # Read noise file. Column order is: ['l', 'Nl', 'Nl+Cl']
            self.noise = np.loadtxt(self.config['file_noise'], unpack=True)

        return self.noise

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
