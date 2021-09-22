from .mapper_base import MapperBase
import healpy as hp


class MapperSPT(MapperBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-ymaps_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_tSZ',
         'nside':512}
        """
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

    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_map, 1)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_map, 2)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_mask(self):
        if self.mask is None:
            mask = hp.read_map(self.file_mask)
            mask = hp.ud_grade(mask,
                               nside_out=self.nside)
            self.mask = mask
        return self.mask
    
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

    def get_dtype(self):
        return 'cmb_tSZ'

    def get_spin(self):
        return 0
