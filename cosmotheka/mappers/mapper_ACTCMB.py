from .mapper_ACT_compsept import MapperACTCompSept


class MapperACTCMB(MapperACTCompSept):
    """
    For X either 'BN' or 'D56' depending on the desired sky patch.

    **Config**

        - mask_name: `'mask_ACT_compsep_X'`
        - map_name: `'compsep_X'`
        - path_rerun: `'.../Datasets/ACT_DR4/xcell_runs'`
        - file_map: `'.../Datasets/ACT_DR4/compsep_maps/\
          tilec_single_tile_X_cmb_map_v1.2.0_joint.fits'`
        - file_mask: `'.../Datasets/ACT_DR4/masks/\
          compsep_masks/act_dr4.01_s14s15_X_compsep_mask.fits'`
        - beam_info:

            - type: `'Custom'`
            - file: `'.../Datasets/ACT_DR4/compsep_maps/\
              tilec_single_tile_X_cmb_map_v1.2.0_joint_beam.txt'`

        - lmax: `6000`
    """
    def get_dtype(self):
        return 'cmb_kSZ'

    def get_spin(self):
        return 0
