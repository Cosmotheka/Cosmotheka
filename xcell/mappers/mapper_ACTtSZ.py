from .mapper_ACT_compsept import MapperACTCompSept


class MapperACTtSZ(MapperACTCompSept):
    """
    For X either 'BN' or 'D56' depending on the desired sky patch.
    
    ***Config***

        - mask_name: `'mask_ACT_compsep_X'`
        - map_name: `'compsep_X'`
        - path_rerun: `'/mnt/extraspace/damonge/Datasets/ACT_DR4/xcell_runs'`
        - file_map: `'/mnt/extraspace/damonge/Datasets/ACT_DR4/compsep_maps/tilec_single_tile_X_comptony_map_v1.2.0_joint.fits'`
        - file_mask: `'/mnt/extraspace/damonge/Datasets/ACT_DR4/masks/compsep_masks/act_dr4.01_s14s15_X_compsep_mask.fits'`
        - beam_info:

           - type: `'Custom'`
           - file: `'/mnt/extraspace/damonge/Datasets/ACT_DR4/compsep_maps/tilec_single_tile_X_comptony_map_v1.2.0_joint_beam.txt'`

        - lmax: `6000`
    """
    def __init__(self, config):
        self._get_ACT_defaults(config)

    def get_dtype(self):
        """
        Returns the type of the mapper. \
        
        Args:
            None
        Returns:
            mapper_type (String)
        """
        return 'cmb_tSZ'

    def get_spin(self):
        """
        Returns the spin of the mapper. \
        
        Args:
            None
        Returns:
            spin (Int)
        """
        return 0
