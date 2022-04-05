from .mapper_base import MapperBase
from pixell import enmap


class MapperACTBase(MapperBase):
    def __init__(self, config):
        self._get_ACT_defaults(config)

    def _get_ACT_defaults(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.file_map = config['file_map']
        self.file_mask = config['file_mask']
        self.map_name = config['map_name']
        self.lmax = config.get('lmax', 6000)
        self.signal_map = None
        self.mask = None
        self.pixell_mask = None
        self.nl_coupled = None

    def get_signal_map(self):
        if self.signal_map is None:
            fn = '_'.join([f'ACT_{self.map_name}_signal',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])
            mp = self._rerun_read_cycle(fn, 'FITSMap', self._get_signal_map)
            self.signal_map = [mp]
        return self.signal_map

    def _get_pixell_mask(self):
        if self.pixell_mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
        return self.pixell_mask

    def get_mask(self):
        if self.mask is None:
            fn = '_'.join([f'ACT_{self.map_name}_mask',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])
            self.mask = self._rerun_read_cycle(fn, 'FITSMap', self._get_mask)
        return self.mask
