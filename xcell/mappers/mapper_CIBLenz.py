import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from .mapper_P15CIB import MapperP15CIB


class MapperCIBLenz(MapperP15CIB):
    """
    Mapper for the P15 CIB data set. \
    Child of Mapperp15CIB. \
    """
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

    def _get_custom_beam(self, info):
        """
        Returns the custom beam of the mapper \
        given an information dictionary. \
        
        Args:
            info (Dict)
        Returns:
            beam (Array)
        
        """
        fname = info['file']
        field = info['field']
        windowfuncs = pd.read_csv(fname, comment='#')
        pixwin = interp1d(np.array(windowfuncs['ell']),
                          np.log(np.array(windowfuncs[field])),
                          fill_value='extrapolate')
        ell = np.arange(3*self.nside)
        return np.exp(pixwin(ell))
