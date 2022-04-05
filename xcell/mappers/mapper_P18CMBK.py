from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt


class MapperP18CMBK(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {'file_klm':'COM_Lensing_4096_R3.00/MV/dat_klm.fits',
         'file_mask':'COM_Lensing_4096_R3.00/mask.fits.gz',
         'file_noise':'COM_Lensing_4096_R3.00/MV/nlkk.dat',
         'mask_aposize': 0.2,
         'mask_apotype': 'C1',
         'mask_name': 'mask_CMBK',
         'nside':4096}
        """
        self._get_defaults(config)
        self.mask_apotype = config.get('mask_apotype', 'C1')
        self.mask_aposize = config.get('mask_aposize', 0.2)

        self.noise = None

        # Galactic-to-celestial coordinate rotator
        self.rot = self._get_rotator('G')

        # Defaults
        self.signal_map = None
        self.nl_coupled = None
        self.mask = None
        self.cl_fid = None

    def get_signal_map(self):
        if self.signal_map is None:
            # Read alms
            self.klm, lmax = hp.read_alm(self.config['file_klm'],
                                         return_mmax=True)
            if self.rot is not None:
                self.klm = self.rot.rotate_alm(self.klm)
            # Filter if lmax is too large
            if lmax > 3*self.nside-1:
                fl = np.ones(lmax+1)
                fl[3*self.nside:] = 0
                self.klm = hp.almxfl(self.klm, fl, inplace=True)
            self.signal_map = np.array([hp.alm2map(self.klm, self.nside)])
        return self.signal_map

    def _get_mask(self):
        msk = hp.read_map(self.config['file_mask'],
                          dtype=float)
        if self.rot is not None:
            msk = self.rot.rotate_map_pixel(msk)
            # Binarize
            thr = self.config.get('mask_threshold', 0.5)
            msk[msk < thr] = 0
            msk[msk >= thr] = 1.
        # Apodize
        msk = nmt.mask_apodization(msk, self.mask_aposize,
                                   self.mask_apotype)
        msk = hp.ud_grade(msk, nside_out=self.nside)
        return msk

    def get_mask(self):
        if self.mask is None:
            fn = '_'.join([f'P18CMBK_mask_{self.mask_aposize}',
                           f'{self.mask_apotype}',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])
            self.mask = self._rerun_read_cycle(fn, 'FITSMap', self._get_mask)
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
            # will be divided for in the covariance.
            nl *= np.mean(self.get_mask()**2.)
            self.nl_coupled = np.array([nl])
        return self.nl_coupled

    def get_cl_fiducial(self):
        if self.cl_fid is None:
            ell = self.get_ell()
            noise = self._get_noise()
            cl = noise[2] - noise[1]
            cl = interp1d(noise[0], cl, bounds_error=False,
                          fill_value=(cl[0], cl[-1]))(ell)
            self.cl_fid = np.array([cl])
        return self.cl_fid

    def _get_noise(self):
        if self.noise is None:
            # Read noise file. Column order is: ['l', 'Nl', 'Nl+Cl']
            self.noise = np.loadtxt(self.config['file_noise'], unpack=True)

        return self.noise

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
