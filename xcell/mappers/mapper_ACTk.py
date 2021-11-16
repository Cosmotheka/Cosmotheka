from .mapper_ACT_base import MapperACTBase
from pixell import enmap, reproject
import numpy as np


class MapperACTk(MapperACTBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map':'act_planck_dr4.01_s14s15_D56_lensing_kappa_baseline.fits',
         'file_mask':'act_dr4.01_s14s15_D56_lensing_mask.fits',
         'mask_name': 'mask_ACTK',
         'nside': 1024,
         'lmax': 6000}
        """
        self._get_ACT_defaults(config)

    def get_signal_map(self):
        if self.signal_map is None:
            self.pixell_mask = self._get_pixell_mask()
            self.signal_map = enmap.read_map(self.file_map)
            self.signal_map = [reproject.healpix_from_enmap(self.signal_map,
                                                            lmax=self.lmax,
                                                            nside=self.nside)]
            self.signal_map *= np.mean(self.pixell_mask**2)
        return self.signal_map

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
