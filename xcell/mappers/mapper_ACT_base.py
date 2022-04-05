from .mapper_base import MapperBase
from pixell import enmap
import healpy as hp


class MapperACTBase(MapperBase):
    def __init__(self, config):
        self._get_ACT_defaults(config)

    def _get_ACT_defaults(self, config):
        self._get_defaults(config)
        if self.coords != 'C':
            self.rot = hp.Rotator(coord=['C', self.coords])
        else:
            self.rot = None
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
            fn = f'ACT_{self.map_name}_coord{self.coord}_signal.fits.gz'
            mp = self._rerun_read_cycle(fn, 'FITSMap', self._get_signal_map)
            self.signal_map = [mp]
        return self.signal_map

    def _get_pixell_mask(self):
        if self.pixell_mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
        return self.pixell_mask

    def get_mask(self):
        if self.mask is None:
            fn = f'ACT_{self.map_name}_coord{self.coord}_mask.fits.gz'
            self.mask = self._rerun_read_cycle(fn, 'FITSMap', self._get_mask)
        return self.mask
