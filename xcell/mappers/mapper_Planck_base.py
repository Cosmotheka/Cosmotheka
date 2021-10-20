import numpy as np
import healpy as hp
import pymaster as nmt
from .mapper_base import MapperBase


class MapperPlanckBase(MapperBase):
    def __init__(self, config):
        self._get_Planck_defaults(config)

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
        self.cls_cov = None
        self.custom_auto = True
        self.mask = None
        self.beam = None
        self.beam_info = None
        self.gal_mask_mode = config.get('gal_mask_mode', '0.6')
        self.gal_mask_modes = {'0.2': 0,
                               '0.4': 1,
                               '0.6': 2,
                               '0.7': 3,
                               '0.8': 4,
                               '0.9': 5,
                               '0.97': 6,
                               '0.99': 7}

    def get_signal_map(self):
        if self.signal_map is None:
            signal_map = hp.read_map(self.file_map)
            self.signal_map = [hp.ud_grade(signal_map,
                                           nside_out=self.nside)]
        return self.signal_map

    def get_mask(self):
        return NotImplementedError("Do not use base class")

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

    def _beam_gaussian(self, ell, fwhm_amin):
        sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
        return np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)

    def get_beam(self):
        if self.beam is None:
            if self.beam_info is None:  # No beam
                self.beam = np.ones(3*self.nside)
            else:
                ell = np.arange(3*self.nside)
                self.beam = self._beam_gaussian(ell, self.beam_info)
                self.beam /= self.beam[0] #Apparently you have to normalize it
        return self.beam

    def get_spin(self):
        return 0
