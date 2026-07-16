from .mapper_Planck_base import MapperPlanckBase
from .utils import rotate_map
import healpy as hp
import numpy as np


class MapperPlanckNPIPE(MapperPlanckBase):
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
    map_name = "PlanckNPIPE"

    def __init__(self, config):
        self._get_Planck_defaults(config)
        self.beam_info = config.get("beam_info", [])
        self.gp_mask_mode = config.get('gp_mask_mode', '0.6')
        self.gp_mask_modes = {'0.2': 0,
                              '0.4': 1,
                              '0.6': 2,
                              '0.7': 3,
                              '0.8': 4,
                              '0.9': 5,
                              '0.97': 6,
                              '0.99': 7}
        self.ps_mask_modes = {'F100': 0,
                              'F143': 1,
                              'F217': 2,
                              'F353': 3}
        self.ps_mask_mode = config.get('ps_mask_mode',
                                       ['F100', 'F143', 'F217', 'F353'])

        self.map_name = f"{self.map_name}_{config['freq']}"
        self.mask_name = f"{self.mask_name}_{self.gp_mask_mode}"

    def _remove_dipole(self, map_in):
        mask = self._get_mask()
        mask[mask != 0] = 1
        map_in *= mask
        map_in[map_in == 0] = hp.UNSEEN
        map_in = hp.remove_dipole(map_in)
        map_in[map_in == hp.UNSEEN] = 0.0
        return map_in

    def _read_and_rotate(self, fname):
        m = hp.read_map(fname)
        ps_mask = self._get_ps_mask()
        nside = hp.npix2nside(len(m))
        ps_mask = hp.ud_grade(ps_mask, nside_out=nside)

        m *= ps_mask
        m = rotate_map(m, self.rot)
        m = hp.ud_grade(m, nside_out=self.nside)
        return m

    def _rotate_map(self, map_in):
        ps_mask = self._get_ps_mask()
        nside = hp.npix2nside(len(map_in))
        ps_mask = hp.ud_grade(ps_mask, nside_out=nside)
        map_in *= ps_mask
        map_in = rotate_map(map_in, self.rot)
        map_in = hp.ud_grade(map_in, nside_out=self.nside)
        return map_in

    def _read_and_remove_dipole(self, fname):
        field = self.gp_mask_modes[self.gp_mask_mode]
        gal_mask = hp.read_map(
            self.file_gp_mask,
            field
        )
        m = hp.read_map(fname)
        nside = hp.npix2nside(len(m))
        gal_mask = hp.ud_grade(gal_mask, nside_out=nside)
        gal_mask[gal_mask != 0] = 1
        m *= gal_mask
        m[m == 0] = hp.UNSEEN
        m = hp.remove_dipole(m)
        m[m == hp.UNSEEN] = 0.0
        return m

    def _generate_hm_maps(self):
        hm1_map = self._read_and_remove_dipole(self.file_hm1)
        hm1_map = self._rotate_map(hm1_map)
        hm2_map = self._read_and_remove_dipole(self.file_hm2)
        hm2_map = self._rotate_map(hm2_map)

        return np.array([hm1_map, hm2_map])

    def _get_signal_map(self):
        signal_map = self._read_and_remove_dipole(self.file_map)
        signal_map = self._rotate_map(signal_map)

        return np.array([signal_map])

    def get_dtype(self):
        """
        Returns the type of the mapper. \

        Args:
            None
        Returns:
            mapper_type (String)
        """
        return 'cmb_temperature'
