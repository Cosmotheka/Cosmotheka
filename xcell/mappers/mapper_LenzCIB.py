from .mapper_P15CIB import MapperP15CIB


class MapperLenzCIB(MapperP15CIB):
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
                                    {'type': 'Gaussian',
                                     'FWHM_arcmin': 5.0})

    def _get_custom_wf(wf_info):
        field = wf_info['field']
        file = wf_info['file']
        windowfuncs = pd.read_csv(file, comment='#')
        wf = interp1d(np.array(windowfuncs['ell']),
                      np.array(windowfuncs[field]),
                      fill_value='extrapolate')
        return wf
