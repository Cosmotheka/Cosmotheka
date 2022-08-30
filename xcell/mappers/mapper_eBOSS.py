from .mapper_SDSS import MapperSDSS
import numpy as np


class MappereBOSS(MapperSDSS):
    """
    Mapper for the eBOSS DR14 data set. \
    Child class of "MapperSDSS". 
    """
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':['eBOSS_QSO_clustering_data-NGC-vDR16.fits'],
           'random_catalogs':['eBOSS_QSO_clustering_random-NGC-vDR16.fits'],
           'z_edges':[0, 1.5],
           'nside':nside,
           'nside_mask': nside_mask,
           'mask_name': 'mask_QSO_NGC_1'}
        """
        self._get_SDSS_defaults(config)
        self.z_edges = config.get('z_edges', [0, 3])

    def _get_w(self, mod='data'):
        """
        Returns the weights for the sources of \
        the mapper's data or randoms catalog. 
        
        Kwargs:
            mod="data"
        Returns:
            ws (Array)
        """
        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            w_systot = np.array(cat['WEIGHT_SYSTOT'])
            w_cp = np.array(cat['WEIGHT_CP'])
            w_noz = np.array(cat['WEIGHT_NOZ'])
            self.ws[mod] = w_systot*w_cp*w_noz  # FKP left out
        return self.ws[mod]

    def _get_random_cols(self):
        """
        Returns the names of the columns \
        of interest for the mapper. 
        
        Args:
            None

        Returns:
            ['RA', 'DEC', 'Z', 'WEIGHT_SYSTOT',
             'WEIGHT_CP', 'WEIGHT_NOZ'] (Array)      
        """
        return ['RA', 'DEC', 'Z', 'WEIGHT_SYSTOT',
                'WEIGHT_CP', 'WEIGHT_NOZ']
