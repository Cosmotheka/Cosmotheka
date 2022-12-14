from .mapper_ACT_base import MapperACTBase
from .utils import rotate_mask, rotate_map
from pixell import enmap, reproject
import numpy as np


class MapperACTk(MapperACTBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map':path_ACT+'act_planck_dr4.01_s14s15_D56_lensing_kappa_baseline.fits',
        'file_mask':path_ACT+'act_dr4.01_s14s15_D56_lensing_mask.fits',
        'mask_name': 'mask_CMBK',
        'nside': nside,
        'mask_power': 2}
        """
        self._get_ACT_defaults(config)
        self.mask_power = config.get('mask_power', 2)
        self.mask_threshold = config.get('mask_threshold', 0.1)

    def _get_signal_map(self):
        self.pixell_mask = self._get_pixell_mask()
        mp = enmap.read_map(self.file_map)
        mp = reproject.healpix_from_enmap(mp,
                                          lmax=self.lmax,
                                          nside=self.nside)
        mp = rotate_map(mp, self.rot)
        mp *= np.mean(self.pixell_mask**2)
        return mp

    def _get_mask(self):
        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.healpix_from_enmap(self.pixell_mask,
                                           lmax=self.lmax,
                                           nside=self.nside)
        msk = rotate_mask(msk, self.rot)
        # Cap it (there is noise due to the change of pixelization)
        goodpix = msk > self.mask_threshold
        msk[~goodpix] = 0
        return msk

    def get_nl_coupled(self):
        # raise NotImplementedError("No noise model for the ACT maps")
        return np.zeros([1, 3*self.nside])

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
