from .mapper_base import MapperBase
from pixell import enmap


class MapperACTBase(MapperBase):
    """
    Base ACT mapper class used as foundation \
    for the rest of ACT mappers.
    """
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
        self.pixell_mask = None
        self.nl_coupled = None

    def get_signal_map(self):
        """
        Returns the signal map of the children \
        mapppers of the base class. \
        If children mappers have already calculated \
        the signal mapper, the function loads it \
        from a file. \
        Otherwise, each mapper's "_get_signal_map()" \
        is used to calculate it.

        Returns:
            delta_map (Array)
        """
        if self.signal_map is None:
            fn = '_'.join([f'ACT_{self.map_name}_signal',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])
            mp = self._rerun_read_cycle(fn, 'FITSMap', self._get_signal_map)
            self.signal_map = [mp]
        return self.signal_map

    def _get_pixell_mask(self):
        """
        Returns the mask of the mapper \
        in pixell format.

        Returns
            pixel_mask (Array)
        """
        if self.pixell_mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
        return self.pixell_mask
