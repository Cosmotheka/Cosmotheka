from .mapper_Planck_base import MapperPlanckBase
from .utils import rotate_map
import healpy as hp


class MapperP18SMICA(MapperPlanckBase):
    """
    path = `".../Datasets/Planck_SMICA"`
    **Config**

        - file_map: `path+"COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"`
        - file_hm1: `path+"COM_CMB_IQU-smica-nosz_2048_R3.00_hm1.fits"`
        - file_hm2: `path+"COM_CMB_IQU-smica-nosz_2048_R3.00_hm2.fits"`
        - file_gp_mask: \
        `".../Datasets/Planck_masks/HFI_Mask_GalPlane-apo2_2048_R2.00.fits"`
        - file_ps_mask: \
        `".../Datasets/Planck_masks/HFI_Mask_PointSrc_2048_R2.00.fits"`

        - beam_info:

            - type: `'Gaussian'`
            - FWHM_arcmin: `5.0`

        - gp_mask_mode: `'0.6'`
        - ps_mask_mode: `['F100', 'F143', 'F217', 'F353']`
        - mask_name: `mask_P18SMICA`
        - path_rerun: `'.../Datasets/Planck_SMICA/xcell_runs'`
    """
    def __init__(self, config):
        self._get_Planck_defaults(config)
        self.beam_info = config.get('beam_info',
                                    [{'type': 'Gaussian',
                                      'FWHM_arcmin': 5.0}])
        self.gp_mask_mode = config.get('gp_mask_mode', '0.6')
        self.gp_mask_modes = {'0.2': 0,
                              '0.4': 1,
                              '0.6': 2,
                              '0.7': 3,
                              '0.8': 4,
                              '0.9': 5,
                              '0.97': 6,
                              '0.99': 7}

    def _get_hm_maps(self):
        """
        Returns the half mission maps of the mapper \
        after applying the \
        neccesary coordinate rotations. \
        Args:
            None
        Returns:
            hm1_map (Array)
            hm2_map (Array)
        """
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1)
            hm1_map = rotate_map(hm1_map, self.rot)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            hm2_map = rotate_map(hm2_map, self.rot)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        """
        Returns the type of the mapper. \

        Args:
            None
        Returns:
            mapper_type (String)
        """
        return 'cmb_temperature'
