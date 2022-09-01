import numpy as np
import healpy as hp
import pymaster as nmt
from .mapper_base import MapperBase
from .utils import rotate_mask, rotate_map


class MapperPlanckBase(MapperBase):
    """
    Base mapper for the Planck mappers.
    """
    def __init__(self, config):
        self._get_Planck_defaults(config)

    def _get_Planck_defaults(self, config):
        # Creates instances of common elements \
        # between the different Planck mappers.

        self._get_defaults(config)
        self.rot = self._get_rotator('G')
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

    def get_signal_map(self):
        if self.signal_map is None:
            signal_map = hp.read_map(self.file_map)
            signal_map[signal_map == hp.UNSEEN] = 0.0
            signal_map[np.isnan(signal_map)] = 0.0
            signal_map = rotate_map(signal_map, self.rot)
            self.signal_map = np.array([hp.ud_grade(signal_map,
                                        nside_out=self.nside)])
        return self.signal_map

    def _get_mask(self):
        # Returns the mask of the mapper. \
        # if the mapper doesn't have a base mask \
        # a full sky mask is created. \
        # If the mapper is equipped with a \
        # galactic plane mask, the galactic plane and \
        # the base masks are multiplied. \
        # If the mapper is equipped with a \
        # point source mask, the point source and \
        # the base masks are multiplied.

        msk = None
        if self.file_mask is not None:
            msk = hp.read_map(self.file_mask)
        if self.file_gp_mask is not None:
            field = self.gp_mask_modes[self.gp_mask_mode]
            gp_mask = hp.read_map(self.file_gp_mask, field)
            if msk is None:
                msk = gp_mask
            else:
                msk *= gp_mask
        if self.file_ps_mask is not None:
            for mode in self.ps_mask_mode:
                field = self.ps_mask_modes[mode]
                ps_mask = hp.read_map(self.file_ps_mask, field)
                if msk is None:
                    msk = ps_mask
                else:
                    msk *= ps_mask
        msk = rotate_mask(msk, self.rot)
        msk[msk < 0] = 0
        msk = hp.ud_grade(msk, nside_out=self.nside)
        return msk

    def _get_hm_maps(self):
        # Returns the half mission maps
        # of the mapper
        return NotImplementedError("Do not use base class")

    def _get_diff_map(self):
        # Substracts the two half mission maps \
        # of the mapper.

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
        """
        Uses the half mission maps to \
        estimate the coupled signal power \
        spectrum of the mapper.

        Returns:
            cl_coupled (Array)
        """
        if self.cl_coupled is None:
            self.hm1_map, self.hm2_map = self._get_hm_maps()
            hm1_f = self._get_nmt_field(signal=self.hm1_map)
            hm2_f = self._get_nmt_field(signal=self.hm2_map)
            self.cl_coupled = nmt.compute_coupled_cell(hm1_f, hm2_f)
        return self.cl_coupled

    def get_cls_covar_coupled(self):
        """
        Uses the half mission maps to \
        estimate the coupled covariance matrix of the \
        power spectrum of the coadded map as \
        well as the half mission maps cross- \
        and auto-correlation.

        Returns:
            cl_coupled (Array)
        """
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
