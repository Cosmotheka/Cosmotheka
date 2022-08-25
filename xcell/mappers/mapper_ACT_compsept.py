from .mapper_ACT_base import MapperACTBase
from .utils import rotate_mask, rotate_map
from pixell import enmap, reproject
import numpy as np
from scipy.interpolate import interp1d


class MapperACTCompSept(MapperACTBase):
    """
    Base mapper class for the ACT \
    component separated data sets. \
    """
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
        self._get_ACT_defaults(config)

    def _get_signal_map(self):
        """
        Returns the signal map of the mappper.
        
        Args:
            None
        Returns:
            delta_map (Array)
        
        """
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
        """
        Returns the mask of the mappper.
        
        Args:
            None
        Returns:
            mask (Array)
        
        """
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
         """
        Returns the custom beam of the mapper \
        given an information dictionary. \
        
        Args:
            info (Dict)
        Returns:
            beam (Array)
        
        """
        fname = info['file']
        beam_file = np.transpose(np.loadtxt(fname))
        ells = beam_file[0]
        beam = beam_file[1]
        pixwin = interp1d(ells, np.log(beam),
                          fill_value='extrapolate')
        ell = np.arange(3*self.nside)
        return np.exp(pixwin(ell))
