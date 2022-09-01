from .mapper_ACT_base import MapperACTBase
from .utils import rotate_mask, rotate_map
from pixell import enmap, reproject
import numpy as np


class MapperACTk(MapperACTBase):
    """
    For X either 'BN' or 'D56' depending on the desired sky patch.

    **Config**

        - mapper_class: `'MapperACTk'`
        - mask_name: `'mask_ACT_kappa_X'`
        - map_name: `'kappa_X'`
        - path_rerun: `'.../Datasets/ACT_DR4/xcell_runs'`
        - file_map: \
        `'.../Datasets/ACT_DR4/lensing_kappa_maps/\
         act_planck_dr4.01_s14s15_X_lensing_kappa_baseline.fits'`
        - file_mask: \
        `'.../Datasets/ACT_DR4/masks/lensing_masks/\
         act_dr4.01_s14s15_X_lensing_mask.fits'`
        - lmax: `6000`
        - mask_power: `2`
    """
    def __init__(self, config):
        self._get_ACT_defaults(config)
        self.mask_power = config.get('mask_power', 2)

    def _get_signal_map(self):
        # Loads the signal_map in pixell format
        # and coverts it to healpy

        self.pixell_mask = self._get_pixell_mask()
        mp = enmap.read_map(self.file_map)
        mp = reproject.healpix_from_enmap(mp,
                                          lmax=self.lmax,
                                          nside=self.nside)
        mp = rotate_map(mp, self.rot)
        mp *= np.mean(self.pixell_mask**2)
        return mp

    def _get_mask(self):
        # Loads the signal_map in pixell format
        # and coverts it to healpy

        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.healpix_from_enmap(self.pixell_mask,
                                           lmax=self.lmax,
                                           nside=self.nside)
        msk = rotate_mask(msk, self.rot)
        return msk

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
