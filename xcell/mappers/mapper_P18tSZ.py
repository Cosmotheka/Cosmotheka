from .mapper_base_Planck import MapperBasePlanck


class MapperP18tSZ(MapperBasePlanck):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-ymaps_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_tSZ',
         'nside':512}
        """
        self._get_Planck_defaults(config)

    def get_dtype(self):
        return 'cmb_tSZ'
