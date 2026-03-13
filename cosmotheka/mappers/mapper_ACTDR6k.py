from .mapper_base import MapperBase
from .utils import rotate_mask, rotate_map
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import warnings
import glob


class MapperACTDR6k(MapperBase):
    """
    ACT DR6 kappa mapper class.

    Information about the data used in this mapper: https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_maps_info.html
    Extended products available at: https://portal.nersc.gov/project/act/dr6_lensing_v1/

    **Config**

        - mask_name: ACT_kappa_DR6
        - map_name: kappa_DR6
        - path_rerun: /mnt/extraspace/damonge/Datasets/ACT_DR6/xcell_runs
        - klm_file: /mnt/extraspace/damonge/Datasets/ACT_DR6/lensing_maps/baseline/kappa_alm_data_act_dr6_lensing_v1_baseline.fits
        - file_mask: /mnt/extraspace/damonge/Datasets/ACT_DR6/lensing_maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits
        - file_noise: /mnt/extraspace/damonge/Datasets/ACT_DR6/lensing_maps/baseline/N_L_kk_act_dr6_lensing_v1_baseline.txt
        - lmax: 4000
        - mask_threshold: 0.1  # Avoid ringing
        - variant: "baseline"
        - sims_rec_path: /mnt/extraspace/damonge/Datasets/ACT_DR6/lensing_maps/baseline/simulations
        - sims_in_path: /mnt/extraspace/damonge/Datasets/ACT_DR6/lensing_maps/baseline/simulations
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

    # TODO: Create a kappa mapers class to avoid repetition.
    def _get_sims_fnames(self):
        """
        Returns the paths of the reconstructed and input simulation maps.

        Returns:
            rec_sims (List): list of paths to reconstructed simulation maps
            input_sims (List): list of paths to input simulation maps
        """
        rec_sims_path = self.config['sims_rec_path']
        input_sims_path = self.config['sims_in_path']

        # Using glob because it's handy. If the naming convention is wrong,
        # this might silently mix rec and input sims and spoil the transfer 
        # function.
        pattern = 'kappa_alm_sim_act_dr6_lensing_v1_baseline_*.fits'
        rec_sims = sorted(glob.glob(rec_sims_path + '/' + pattern))
        pattern = 'input_kappa_alm_sim_*.fits'
        input_sims = sorted(glob.glob(input_sims_path + '/' + pattern))

        nrec = len(rec_sims)
        ninput = len(input_sims)

        print(f"Found {nrec} reconstructed sims and {ninput} input sims")

        # Check that there are the same number of sims
        if len(rec_sims) != len(input_sims):
            raise ValueError("Number of reconstructed and input sims must be "
                             "the same. Found {nrec} reconstructed and "
                             "{ninput} input sims.")

        return rec_sims, input_sims