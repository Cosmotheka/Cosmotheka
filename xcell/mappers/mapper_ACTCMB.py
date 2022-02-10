from .mapper_ACT_base import MapperACTBase
from pixell import enmap, reproject


from .mapper_ACT_compsept import MapperACTCompSept
from pixell import enmap, reproject


class MapperACTCMB(MapperACTCompSept):
    def __init__(self, config):
        """
        config - dict
        {'file_map':'tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits',
         'file_mask':'tilec_mask.fits',
         'file_noise': 'tilec_single_tile_D56_cmb_map_v1.2.0_joint_noise.fits',
         'beam_file': 'tilec_single_tile_D56_cmb_map_v1.2.0_joint_beam.txt',
         'mask_name': 'mask_ACTtsz',
         'nside': 1024,
         'lmax': 6000}
        """
        self._get_ACT_CompSept_defaults(config)

    def get_dtype(self):
        return 'cmb_kSZ'

    def get_spin(self):
        return 0
