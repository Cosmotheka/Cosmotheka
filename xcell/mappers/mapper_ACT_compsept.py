from .mapper_ACT_base import MapperACTBase
from .utils import rotate_mask, rotate_map
from pixell import enmap, reproject
import numpy as np
from scipy.interpolate import interp1d


class MapperACTCompSept(MapperACTBase):
    """
    Base mapper class for the ACT \
    component separated mappers. \
    """
    def __init__(self, config):
        self._get_ACT_defaults(config)

    def _get_signal_map(self):
        # Loads pixell map and converts it
        # to healpy

        if self.signal_map is None:
            # The 'Weights' FITS file contains the 2D Fourier space
            # weight for each pixel corresponding to the detector array.
            signal_map = enmap.read_map(self.file_map)
            signal_map = reproject.healpix_from_enmap(signal_map,
                                                      lmax=self.lmax,
                                                      nside=self.nside)
            signal_map = rotate_map(signal_map, self.rot)
        return signal_map

    def _get_mask(self):
        # It [The mask] has already been applied to the map.
        # If you are doing a real-space analysis, you should
        # exclude any pixels where the value in the mask is
        # appreciably different from 1 since the signal there
        # should be attenuated by the value in the mask.

        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.healpix_from_enmap(self.pixell_mask,
                                           lmax=self.lmax,
                                           nside=self.nside)
        msk[msk < 0.99] = 0
        msk = rotate_mask(msk, self.rot)
        return msk

    def get_nl_coupled(self):
        raise NotImplementedError("No noise model for the ACT maps")

    def _get_custom_beam(self, info):
        # Loads beam from file

        fname = info['file']
        beam_file = np.transpose(np.loadtxt(fname))
        ells = beam_file[0]
        beam = beam_file[1]
        pixwin = interp1d(ells, np.log(beam),
                          fill_value='extrapolate')
        ell = np.arange(3*self.nside)
        return np.exp(pixwin(ell))
