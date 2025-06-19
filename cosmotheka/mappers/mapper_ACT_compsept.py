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
        if self.lmax > 3 * self.nside:
            # WARNING
            print("WARNING:              you selected lmax > 3*nside, "
                  "setting lmax to 3 * nside.")
            self.lmax = 3 * self.nside

    def _get_signal_map(self):
        # The 'Weights' FITS file contains the 2D Fourier space
        # weight for each pixel corresponding to the detector array.
        signal_map = enmap.read_map(self.file_map)
        signal_map = reproject.map2healpix(
            signal_map,
            nside=self.nside,
            lmax=self.lmax,
            niter=3
        )
        signal_map = rotate_map(signal_map, self.rot)
        return signal_map

    def _get_mask(self):
        # It [The mask] has already been applied to the map.
        # If you are doing a real-space analysis, you should
        # exclude any pixels where the value in the mask is
        # appreciably different from 1 since the signal there
        # should be attenuated by the value in the mask.

        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.map2healpix(
            self.pixell_mask,
            nside=self.nside,
            lmax=self.lmax,
            method="spline",
            order=1
        )
        msk[msk < 0.99] = 0
        msk = rotate_mask(msk, self.rot)
        return msk

    def get_nl_coupled(self):
        # raise NotImplementedError("No noise model for the ACT maps")
        return np.zeros([1, 3*self.nside])

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
