from .mapper_P18CMBK import MapperP18CMBK
import numpy as np


class MapperPlanckPR4CMBK(MapperP18CMBK):
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

    """
    map_name = 'PlanckPR4CMBK'

    def get_cl_fiducial(self):
        raise NotImplementedError("Fiducial signal C_ell not provided for "
                                  "PR4 CMB lensing map.")

    def _get_noise(self):
        # Returns the decoupled noise power spectrum of the \
        # auto-correlation of the covergence map.

        # Returns:
        #     [l (Array): multipole list,
        #      Nl (Array): noise power spectrum] (Array)

        if self.noise is None:
            # Read noise file. Single column with Nl at integer ells
            nl = np.loadtxt(self.config['file_noise'])
            # Zero nan
            nl[0] = 0
            ls = np.arange(len(nl))
            self.noise = (ls, nl)

        return self.noise
