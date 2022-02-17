from .mapper_base import MapperBase
from pixell import enmap, reproject


class MapperACTBase(MapperBase):
    """ Mother class for the different ACT mapper.
    """
    def __init__(self, config):
        self._get_ACT_defaults(config)

    def _get_ACT_defaults(self, config):
        """ Input: configuration dictionary.
            Output: None
            Fills in the shared objects between 
            ACT daughter classes with the information
            from the configuration dictionary.
            Notably, ACT uses the Pixell library for
            its products.
            It is useful to save the mask in Pixeell
            format.
        """
        self._get_defaults(config)
        self.file_map = config['file_map']
        self.file_mask = config['file_mask']
        self.map_name = config['map_name']
        self.lmax = config.get('lmax', 6000)
        self.signal_map = None
        self.mask = None
        self.pixell_mask = None

    def get_signal_map(self):
        return NotImplementedError("Do not use base class")

    def _get_pixell_mask(self):
        """ Input: None
            Output: mapper mask in Pixell format
        """
        if self.pixell_mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
        return self.pixell_mask

    def _get_mask(self):
        """ Input: None
            Output: mapper mask in healpy format
        """
        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.healpix_from_enmap(self.pixell_mask,
                                           lmax=self.lmax,
                                           nside=self.nside)
        return msk

    def get_mask(self):
        """ Input: None
            Output: if the mapper has been previously used
            returns the rerun file for the mask. 
            Otherwise, it calculates the mask and returns it.
        """
        if self.mask is None:
            fn = f'ACT_{self.map_name}_mask.fits.gz'
            self.mask = self._rerun_read_cycle(fn, 'FITSMap', self._get_mask)
        return self.mask
