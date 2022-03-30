import numpy as np
import healpy as hp
import pymaster as nmt
from .mapper_base import MapperBase
from .utils import rotate_mask, rotate_map


class MapperPlanckBase(MapperBase):
    def __init__(self, config):
        self._get_Planck_defaults(config)

    def _get_Planck_defaults(self, config):
        self._get_defaults(config)
        if self.coords != 'G':
            self.rot = hp.Rotator(coord=['G', self.coords])
        else:
            self.rot = None
        self.file_map = config['file_map']
        self.file_hm1 = config.get('file_hm1', None)
        self.file_hm2 = config.get('file_hm2', None)
        self.file_mask = config.get('file_mask', None)
        self.file_gp_mask = config.get('file_gp_mask', None)
        self.file_ps_mask = config.get('file_ps_mask', None)
        self.signal_map = None
        self.hm1_map = None
        self.hm2_map = None
        self.diff_map = None
        self.nl_coupled = None
        self.cl_coupled = None
        self.cls_cov = None
        self.custom_auto = True
        self.mask = None

    def get_signal_map(self):
        if self.signal_map is None:
            signal_map = hp.read_map(self.file_map)
            signal_map[signal_map == hp.UNSEEN] = 0.0
            signal_map[np.isnan(signal_map)] = 0.0
            signal_map = rotate_map(signal_map, self.rot)
            self.signal_map = np.array([hp.ud_grade(signal_map,
                                        nside_out=self.nside)])
        return self.signal_map

    def get_mask(self):
        if self.mask is None:
            if self.file_mask is not None:
                self.mask = hp.read_map(self.file_mask)
            else:
                self.mask = np.ones(12*self.nside**2)
            if self.file_gp_mask is not None:
                field = self.gp_mask_modes[self.gp_mask_mode]
                gp_mask = hp.read_map(self.file_gp_mask, field)
                self.mask *= gp_mask
            if self.file_ps_mask is not None:
                for mode in self.ps_mask_mode:
                    field = self.ps_mask_modes[mode]
                    ps_mask = hp.read_map(self.file_ps_mask, field)
                    self.mask *= ps_mask
            self.mask = rotate_mask(self.mask, self.rot)  # Binarize?
            self.mask = hp.ud_grade(self.mask,
                                    nside_out=self.nside)
        return self.mask

    def _get_hm_maps(self):
        return NotImplementedError("Do not use base class")

    def _get_diff_map(self):
        if self.diff_map is None:
            self.hm1_map, self.hm2_map = self._get_hm_maps()
            self.diff_map = [(self.hm1_map[0] - self.hm2_map[0])/2]
        return self.diff_map

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.diff_map = self._get_diff_map()
            diff_f = self._get_nmt_field(signal=self.diff_map)
            self.nl_coupled = nmt.compute_coupled_cell(diff_f, diff_f)
        return self.nl_coupled

    def get_cl_coupled(self):
        if self.cl_coupled is None:
            self.hm1_map, self.hm2_map = self._get_hm_maps()
            hm1_f = self._get_nmt_field(signal=self.hm1_map)
            hm2_f = self._get_nmt_field(signal=self.hm2_map)
            self.cl_coupled = nmt.compute_coupled_cell(hm1_f, hm2_f)
        return self.cl_coupled

    def get_cls_covar_coupled(self):
        if self.cls_cov is None:
            self.signal_map = self.get_signal_map()
            self.hm1_map, self.hm2_map = self._get_hm_maps()
            coadd_f = self._get_nmt_field(signal=self.signal_map)
            hm1_f = self._get_nmt_field(signal=self.hm1_map)
            hm2_f = self._get_nmt_field(signal=self.hm2_map)
            cl_cc = nmt.compute_coupled_cell(coadd_f, coadd_f)
            cl_11 = nmt.compute_coupled_cell(hm1_f, hm1_f)
            cl_12 = nmt.compute_coupled_cell(hm1_f, hm2_f)
            cl_22 = nmt.compute_coupled_cell(hm2_f, hm2_f)
            self.cls_cov = {'cross': cl_cc,
                            'auto_11': cl_11,
                            'auto_12': cl_12,
                            'auto_22': cl_22}
        return self.cls_cov

    def get_spin(self):
        return 0
