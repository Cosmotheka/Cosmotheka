import numpy as np
import healpy as hp
import pymaster as nmt
from .mapper_base import MapperBase


class MapperBasePlanck(MapperBase):
    def __init__(self, config):
        self._get_Planck_defaults(config)
        return

    def _get_Planck_defaults(self, config):
        self._get_defaults(config)
        self.file_map = config['file_map']
        self.file_hm1 = config.get('file_hm1', None)
        self.file_hm2 = config.get('file_hm2', None)
        self.file_mask = config.get('file_mask', None)
        self.file_gp_mask = config.get('file_gp_mask', None)
        self.file_sp_mask = config.get('file_sp_mask', None)
        self.signal_map = None
        self.hm1_map = None
        self.hm2_map = None
        self.diff_map = None
        self.nl_coupled = None
        self.cl_coupled = None
        self.cls = {'Auto': None,
                    'Cross': None}
        self.custom_auto = True
        self.mask = None
        self.gal_mask_mode = config.get('gal_mask_mode', '60%')
        self.gal_mask_modes = {'20%': 0,
                               '40%': 1,
                               '60%': 2,
                               '70%': 3,
                               '80%': 4,
                               '90%': 5,
                               '97%': 6,
                               '99%': 7}
        self.ell_arr = None
        self.bands = None
        return

    def _get_bands(self):
        i = 0
        ells = []
        while i <= 3 * self.nside:
            ells.append(round(i))
            i = i+20*(1+i/240)

        self.bands = nmt.NmtBin.from_edges(ells[:-1], ells[1:])
        self.ell_arr = self.bands.get_effective_ells()
        return

    def get_signal_map(self):
        if self.signal_map is None:
            signal_map = hp.read_map(self.file_map)
            self.signal_map = [hp.ud_grade(signal_map,
                                           nside_out=self.nside)]
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            self.mask = np.ones(12*self.nside**2)
            if self.file_mask is not None:
                mask = hp.read_map(self.file_mask)
                mask = hp.ud_grade(mask,
                                   nside_out=self.nside)
                self.mask *= mask
            if self.file_gp_mask is not None:
                field = self.gal_mask_modes[self.gal_mask_mode]
                mask = hp.read_map(self.file_gp_mask, field)
                mask = hp.ud_grade(mask,
                                   nside_out=self.nside)
                self.mask *= mask
            if self.file_sp_mask is not None:
                mask = hp.read_map(self.file_sp_mask)
                mask = hp.ud_grade(mask,
                                   nside_out=self.nside)
                self.mask *= mask
        return self.mask

    def _get_hm_maps(self):
        if self.hm1_map is None:
            if self.file_hm1 is not None:
                hm1_map = hp.read_map(self.file_hm1)
                self.hm1_map = [hp.ud_grade(hm1_map,
                                nside_out=self.nside)]
            else:
                hm1_map = hp.read_map(self.file_map, 1)
                self.hm1_map = [hp.ud_grade(hm1_map,
                                nside_out=self.nside)]
        if self.hm2_map is None:
            if self.file_hm2 is not None:
                hm2_map = hp.read_map(self.file_hm2)
                self.hm2_map = [hp.ud_grade(hm2_map,
                                nside_out=self.nside)]
            else:
                hm2_map = hp.read_map(self.file_map, 2)
                self.hm2_map = [hp.ud_grade(hm2_map,
                                nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def _get_diff_map(self):
        if self.diff_map is None:
            self.hm1_map, self.hm2_map = self._get_hm_maps()
            self.diff_map = self.hm1_map[0] - self.hm2_map[0]
        return [self.diff_map]

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.diff_map = self._get_diff_map()
            diff_f = self._get_nmt_field(signal=self.diff_map)
            self.nl_coupled = nmt.compute_coupled_cell(diff_f, diff_f)/4
        return self.nl_coupled

    def get_cl_coupled(self, mode='Auto'):
        self.cl_coupled = self.cls[mode]
        if self.cl_coupled is None:
            if mode == 'Auto':
                f = self.get_nmt_field()
                self.cls[mode] = nmt.compute_coupled_cell(f, f)
            if mode == 'Cross':
                self.hm1_map, self.hm2_map = self._get_hm_maps()
                hm1_f = self._get_nmt_field(signal=self.hm1_map)
                hm2_f = self._get_nmt_field(signal=self.hm2_map)
                self.cls[mode] = nmt.compute_coupled_cell(hm1_f, hm2_f)
            self.cl_coupled = self.cls[mode]
        return self.cl_coupled

    def get_beam(self):
        return self.beam

    def get_spin(self):
        return 0
