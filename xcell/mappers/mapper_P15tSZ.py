from .mapper_Planck_base import MapperPlanckBase
import healpy as hp


class MapperP15tSZ(MapperPlanckBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-ymaps_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_tSZ',
         'nside':512}
        """
        self._get_Planck_defaults(config)
        self.file_hm1 = config.get('hm1_file', self.file_map)
        self.file_hm2 = config.get('hm2_file', self.file_map)
        self.beam_info = config.get('beam_fwhm_arcmin', 10.)
        self.gal_mask_mode = config.get('gal_mask_mode', '1')
        #What are the % of these modes
        self.gal_mask_modes = {'0': 0,
                               '1': 1,
                               '2': 2,
                               '3': 3}

    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1, 1)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2, 2)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_mask(self):
        if self.mask is None:
            field = self.gal_mask_modes[self.gal_mask_mode]
            gal_mask = hp.read_map(self.file_mask, field)
            sp_mask = hp.read_map(self.file_mask, 4)
            mask = gal_mask*sp_mask
            mask = hp.ud_grade(mask,
                               nside_out=self.nside)
            self.mask = mask
        return self.mask

    def get_dtype(self):
        return 'cmb_tSZ'
