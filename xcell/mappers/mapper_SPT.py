from .mapper_Planck_base import MapperPlanckBase
from .utils import rotate_map
import healpy as hp
import numpy as np


class MapperSPT(MapperPlanckBase):
    """
    path = `.../Datasets/SPT/sptsz_planck_ymap_healpix/"`
    **Config**

        - file_map: `path+"SPTSZ_Planck_min_variance_ymap.fits.gz"`
        - file_hm1: `path+"SPTSZ_Planck_min_variance_ymap_half1.fits.gz"`
        - file_hm2: `path+"SPTSZ_Planck_min_variance_ymap_half2.fits.gz"`
        - file_gp_mask: `path+"SPTSZ_dust_mask_top_5percent.fits.gz"`
        - gp_mask_mode: `"default"`
        - file_ps_mask: `path+ \
          "SPTSZ_point_source_mask_nside_8192_binary_mask.fits.gz"`
        - ps_mask_mode: `['default']`

        - beam_info:
            - type: `"Gaussian"`
            - FWHM_arcmin: `1.6`

        - mask_name: `"mask_SPT"`
        - path_rerun: `".../Datasets/SPT/xcell_runs"`
    """
    def __init__(self, config):
        self._get_Planck_defaults(config)
        self.gp_mask_modes = {'default': 0}
        self.gp_mask_mode = config.get('gp_mask_mode', 'default')
        self.ps_mask_modes = {'default': 0}
        self.ps_mask_mode = config.get('ps_mask_mode', ['default'])
        # Fix rotation from Planck's default
        self.rot = self._get_rotator('C')

    def _get_hm_maps(self):
        """
        Returns the half mission maps of the mapper \
        after masking NaN values and applying the \
        neccesary coordinate rotations.

        Returns:
            hm1_map (Array)
            hm2_map (Array)
        """
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1)
            hm1_map[hm1_map == hp.UNSEEN] = 0.0
            hm1_map[np.isnan(hm1_map)] = 0.0
            hm1_map = rotate_map(hm1_map, self.rot)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            hm2_map[hm2_map == hp.UNSEEN] = 0.0
            hm2_map[np.isnan(hm2_map)] = 0.0
            hm2_map = rotate_map(hm2_map, self.rot)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        return 'cmb_tSZ'
