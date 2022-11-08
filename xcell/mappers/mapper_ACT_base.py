from .mapper_base import MapperBase
from pixell import enmap


class MapperACTBase(MapperBase):
    # For backwards compatibility, this mapper allows to pass a 'map_name'
    # configuration argument that will be appended to this map_name
    map_name = 'ACT'

    def __init__(self, config):
        self._get_ACT_defaults(config)

    def _get_ACT_defaults(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.file_map = config['file_map']
        self.file_mask = config['file_mask']
        self.map_name += '_' + config['map_name']
        self.lmax = config.get('lmax', 6000)
        self.pixell_mask = None
        self.nl_coupled = None

    def _get_pixell_mask(self):
        if self.pixell_mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
        return self.pixell_mask
