from .mapper_base_Planck import MapperBasePlanck


class MapperP18SMICA_NOSZ(MapperBasePlanck):
    def __init__(self, config):
        """
        config - dict
        {'file_map': [path+'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits'],
         '',
         'nside':512}
        """
        self._get_Planck_defaults(config)

    def get_dtype(self):
        return 'cmb_SMICA_NOSZ'
