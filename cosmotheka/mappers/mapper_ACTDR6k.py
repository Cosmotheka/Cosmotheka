from .mapper_base import MapperBase
from .utils import rotate_mask, rotate_map
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import warnings


class MapperACTDR6k(MapperBase):
    """
    ACT DR6 kappa mapper class.
    """

    map_name = "ACT"

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator("C")

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

        self.file_noise = config.get("file_noise", None)
        if self.file_noise is None:
            raise ValueError("file_noise must be provided in the config")
        self.file_noise = self.file_noise.replace("baseline", self.variant)
        self.nl_coupled = None

        self.map_name = f"{self.map_name}_{config['map_name']}_{self.variant}"
        self.mask_name = f"{config['mask_name']}_{self.variant}"

        self.lmax = config.get("lmax", 4000)
        warnings.warn(
            f"lmax is set to {self.lmax} but ACT DR6"
            "kappa maps are bandlimited to lmax=4000"
        )

    def _get_signal_map(self):
        return self._get_map_from_klm_file(self.klm_file)

    def _get_mask(self):
        mask = hp.read_map(self.file_mask)
        mask = hp.ud_grade(mask, nside_out=self.nside)
        mask = rotate_mask(mask, self.rot)
        goodpix = mask > self.mask_threshold
        mask[~goodpix] = 0

        return mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            ell, nl = np.loadtxt(self.file_noise, unpack=True)
            nl = interp1d(
                ell, nl, bounds_error=False, fill_value=(nl[0], nl[-1])
            )(self.get_ell())
            # Rescale to "couple" noise
            nl *= np.mean(self.get_mask() ** 2)
            self.nl_coupled = np.array([nl])
        return self.nl_coupled

    def get_dtype(self):
        return "cmb_convergence"

    def get_spin(self):
        return 0
