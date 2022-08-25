from .mapper_ACT_base import MapperACTBase
from .utils import rotate_mask, rotate_map
from pixell import enmap, reproject
import numpy as np


class MapperACTk(MapperACTBase):
    """
    Mapper for the ACT lensing data set. \
    Child class of "MapperACTBase". \
    """
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

    def _get_signal_map(self):
        """
        Returns the signal map of the mappper. \
        
        Args:
            None
        Kwargs:
            apply_galactic_correction=True
        Returns:
            delta_map (Array) 
        """
        self.pixell_mask = self._get_pixell_mask()
        mp = enmap.read_map(self.file_map)
        mp = reproject.healpix_from_enmap(mp,
                                          lmax=self.lmax,
                                          nside=self.nside)
        mp = rotate_map(mp, self.rot)
        mp *= np.mean(self.pixell_mask**2)
        return mp

    def _get_mask(self):
        """
        Returns the mask of the mapper. \
        
        Args:
            None
        Returns:
            mask (Array)
        """
        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.healpix_from_enmap(self.pixell_mask,
                                           lmax=self.lmax,
                                           nside=self.nside)
        msk = rotate_mask(msk, self.rot)
        return msk

    def get_dtype(self):
        """
        Returns the type of the mapper. \
        
        Args:
            None
        Returns:
            mapper_type (String)
        """
        return 'cmb_convergence'

    def get_spin(self):
        """
        Returns the spin of the mapper. \
        
        Args:
            None
        Returns:
            spin (Int)
        """
        return 0
