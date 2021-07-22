from .mapper_base_Planck import MapperBasePlanck


class MapperP15CIB(MapperBasePlanck):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-milca_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_CIB',
         'nside':512}
        """
        self._get_Planck_defaults(config)

    def get_dtype(self):
        return 'cmbCluster_convergenceDensity_cl'
