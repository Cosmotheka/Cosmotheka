from .mapper_ACT_base import MapperACTBase
from .utils import rotate_mask, rotate_map
import numpy as np
import healpy as hp
import warnings


class MapperACTDR6k(MapperACTBase):
    """
    ACT DR6 kappa mapper class.
    """
    def __init__(self, config):
        self._get_defaults(config)

        self.mask_threshold = config.get("mask_threshold", 0.1)
        self.variant = config.get("variant", "baseline")

        self.klm_file = config.get("klm_file", None)
        if self.klm_file is None:
            raise ValueError("klm_file must be provided in the config")
        self.file_mask = config.get("file_mask", None)
        if self.file_mask is None:
            raise ValueError("file_mask must be provided in the config")

        self.klm_file = self.klm_file.replace("baseline", self.variant)
        self.file_mask = self.file_mask.replace("baseline", self.variant)

        self.map_name = f"{self.map_name}_{config['map_name']}_{self.variant}"
        self.mask_name = f"{config['mask_name']}_{self.variant}"

        self.lmax = config.get('lmax', 4000)
        warnings.warn(
            f"lmax is set to {self.lmax} but ACT DR6"
            "kappa maps are bandlimited to lmax=4000"
        )

    def _get_signal_map(self):
        klm, mmax = hp.read_alm(self.klm_file, return_mmax=True)
        klm = klm.astype(np.complex128)
        klm = np.nan_to_num(klm)

        fl = np.ones(mmax + 1)
        fl[3*self.nside:] = 0
        hp.almxfl(klm, fl, inplace=True)

        map = hp.alm2map(klm, nside=self.nside)
        map = rotate_map(map, self.rot)

        mask = self._get_mask()
        map *= np.mean(mask**2)

        return map

    def _get_mask(self):
        mask = hp.read_map(self.file_mask)
        mask = hp.ud_grade(mask, nside_out=self.nside)
        mask = rotate_mask(mask, self.rot)
        goodpix = mask > self.mask_threshold
        mask[~goodpix] = 0

        return mask

    def get_nl_coupled(self):
        # raise NotImplementedError("No noise model for the ACT maps")
        return np.zeros([1, 3*self.nside])

    def get_dtype(self):
        return 'cmb_convergence'

    def get_spin(self):
        return 0
