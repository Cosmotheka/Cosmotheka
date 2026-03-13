import glob

from .mapper_base import MapperBase
from scipy.interpolate import interp1d
import numpy as np
import healpy as hp
import pymaster as nmt
from .utils import rotate_mask


class MapperP18CMBK(MapperBase):
    """
    Note that this mapper is a child of `MapperBase`, /
    not of `MapperPlanckBase`.

    **Config**

        - file_klm: `".../Datasets/Planck_lensing/Lensing2018/MV/dat_klm.fits"`
        - file_mask: `".../Datasets/Planck_lensing/Lensing2018/mask.fits.gz"`
        - file_noise: `".../Datasets/Planck_lensing/Lensing2018/MV/nlkk.dat"`
        - mask_aposize: `0.2`
        - mask_apotype: `"C12"`
        - mask_name: `"mask_P18kappa"`
        - path_rerun: `".../Datasets/Planck_lensing/Lensing2018/xcell_runs"`
        - sims_rec_path = '/mnt/extraspace/vonhausegger/Datasets/Planck_lensing/COM_Lensing-SimMap_4096_R3.00'
        - sims_in_path = '/mnt/extraspace/vonhausegger/Datasets/Planck_lensing/COM_Lensing-SimMap-inputs_4096_R3.00/'

    """
    map_name = 'P18CMBK'

    def __init__(self, config):
        self._get_defaults(config)
        self.mask_apotype = config.get('mask_apotype', 'C1')
        self.mask_aposize = config.get('mask_aposize', 0.2)

        self.noise = None

        # Galactic-to-celestial coordinate rotator
        self.rot = self._get_rotator('G')

        # Defaults
        self.nl_coupled = None
        self.cl_fid = None

    def _get_signal_map(self):
        return self._get_map_from_klm_file(self.config['file_klm'])

    def _get_mask(self):
        msk = hp.read_map(self.config['file_mask'],
                          dtype=float)
        msk = rotate_mask(msk, self.rot, binarize=True)
        # Apodize
        msk = nmt.mask_apodization(msk, self.mask_aposize,
                                   self.mask_apotype)
        msk = hp.ud_grade(msk, nside_out=self.nside)
        return msk

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
        """
        Returns the signal power spectrum \
        of the mapper.

        Returns:
            cl_fid (Array)
        """
        if self.cl_fid is None:
            ell = self.get_ell()
            noise = self._get_noise()
            cl = noise[2] - noise[1]
            cl = interp1d(noise[0], cl, bounds_error=False,
                          fill_value=(cl[0], cl[-1]))(ell)
            self.cl_fid = np.array([cl])
        return self.cl_fid

    def _get_noise(self):
        # Returns the decoupled noise power spectrum of the \
        # auto-correlation of the covergence map.

        # Returns:
        #     [l (Array): multipole list,
        #      Nl (Array): noise power spectrum,
        #      Nl+Cl (Array): noise + signal power spectrum] (Array)

        if self.noise is None:
            # Read noise file. Column order is: ['l', 'Nl', 'Nl+Cl']
            self.noise = np.loadtxt(self.config['file_noise'], unpack=True)

        return self.noise

    def get_dtype(self):
        return "cmb_convergence"

    def get_spin(self):
        return 0

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
        rec_sims = sorted(glob.glob(rec_sims_path + '/' + 'sim_klm_*.fits'))
        input_sims = sorted(glob.glob(input_sims_path + '/' + 'sky_klm_*.fits'))

        nrec = len(rec_sims)
        ninput = len(input_sims)

        print(f"Found {nrec} reconstructed sims and {ninput} input sims")

        # Check that there are the same number of sims
        if len(rec_sims) != len(input_sims):
            raise ValueError("Number of reconstructed and input sims must be "
                             "the same. Found {nrec} reconstructed and "
                             "{ninput} input sims.")

        return rec_sims, input_sims