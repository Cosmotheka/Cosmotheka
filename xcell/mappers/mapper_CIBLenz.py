import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import interp1d
from .utils import rotate_map
from .mapper_Planck_base import MapperPlanckBase


class MapperCIBLenz(MapperPlanckBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-milca_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_CIB',
         'nside':512}
        """
        self._get_Planck_defaults(config)
        self.beam_info = config.get('beam_info',
                                    [{'type': 'Gaussian',
                                      'FWHM_arcmin': 5.0}])
        self.pre_mask_ps = config.get("pre_mask_ps", True)

    def _get_custom_beam(self, info):
        fname = info['file']
        field = info['field']
        windowfuncs = pd.read_csv(fname, comment='#')
        pixwin = interp1d(np.array(windowfuncs['ell']),
                          np.log(np.array(windowfuncs[field])),
                          fill_value='extrapolate')
        ell = np.arange(3*self.nside)
        return np.exp(pixwin(ell))

    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1)
            hm1_map[hm1_map == hp.UNSEEN] = 0.0
            hm1_map[np.isnan(hm1_map)] = 0.0
            hm1_map = rotate_map(hm1_map, self.rot)
            if self.pre_mask_ps is True:
                if self.file_ps_mask is not None:
                    ps_mask = self._get_ps_mask()
                    hm1_map *= ps_mask
                else:
                    NotImplementedError("""Tried to pre-mask point
                                        sources but couldn't find
                                        file_ps_mask""")
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            hm2_map[hm2_map == hp.UNSEEN] = 0.0
            hm2_map[np.isnan(hm2_map)] = 0.0
            hm2_map = rotate_map(hm2_map, self.rot)
            if self.pre_mask_ps is True:
                if self.file_ps_mask is not None:
                    ps_mask = self._get_ps_mask()
                    hm2_map *= ps_mask
                else:
                    NotImplementedError("""Tried to pre-mask point
                                        sources but couldn't find
                                        file_ps_mask""")
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
    def get_dtype(self):
        return 'generic'

