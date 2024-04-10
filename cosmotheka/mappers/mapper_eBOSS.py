from .mapper_SDSS import MapperSDSS
import numpy as np


class MappereBOSS(MapperSDSS):
    """
    Note that X in file names stands for either `'ELG'` or `'LRG'`.

    **Config**

        - num_z_bins: `50`
        - nside_mask: `512`
        - nside_nl_threshold: `4096`
        - lmin_nl_from_data: `2000`
        - data_catalogs: \
        `['.../Datasets/eBOSS/eBOSS_X_clustering_data-NGC-vDR16.fits', \
         '.../Datasets/eBOSS/eBOSS_X_clustering_data-SGC-vDR16.fits']`
        - random_catalogs: \
        `['.../Datasets/eBOSS/eBOSS_X_clustering_random-NGC-vDR16.fits', \
          '.../Datasets/eBOSS/eBOSS_X_clustering_random-SGC-vDR16.fits']`
        - z_edges: `[0.6, 1.1]` / `[0.6, 1.0]`
        - path_rerun: `'.../Datasets/eBOSS/xcell_runs'`
        - mask_name: `'mask_X'`
        - mapper_class: `'MappereBOSS'`
        - bias:  `1.45` / `2.3`
    """
    def __init__(self, config):
        self._get_SDSS_defaults(config)
        self.z_edges = config.get('z_edges', [0, 3])

    def _get_w(self, mod='data'):
        # Returns the weights for the sources of \
        # the mapper's data or randoms catalog.

        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            w_systot = np.array(cat['WEIGHT_SYSTOT'])
            w_cp = np.array(cat['WEIGHT_CP'])
            w_noz = np.array(cat['WEIGHT_NOZ'])
            self.ws[mod] = w_systot*w_cp*w_noz  # FKP left out
        return self.ws[mod]

    def _get_random_cols(self):
        # Returns the names of the columns \
        # of interest for the mapper.
        return ['RA', 'DEC', 'Z', 'WEIGHT_SYSTOT',
                'WEIGHT_CP', 'WEIGHT_NOZ']
