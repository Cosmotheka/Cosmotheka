from .mapper_ACT_base import MapperACTBase
from pixell import enmap, reproject
import numpy as np
from scipy.interpolate import interp1d


class MapperACTCompSept(MapperACTBase):
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

    def _get_ACT_CompSept_defaults(self, config):
        self._get_defaults(config)
        self._get_ACT_defaults(config)
        self.nl_coupled = None
        # self.beam_info = config.get('beam_info',
        #                            [{'type': 'custom',
        #                              'file': None}])

    def _get_signal_map(self):
        if self.signal_map is None:
            # The 'Weights' FITS file contains the 2D Fourier space
            # weight for each pixel corresponding to the detector array.
            signal_map = enmap.read_map(self.file_map)
            signal_map = reproject.healpix_from_enmap(signal_map,
                                                      lmax=self.lmax,
                                                      nside=self.nside)
        return signal_map

    def _get_mask(self):
        if self.mask is None:
            # It [The mask] has already been applied to the map.
            # If you are doing a real-space analysis, you should
            # exclude any pixels where the value in the mask is
            # appreciably different from 1 since the signal there
            # should be attenuated by the value in the mask.

            self.pixell_mask = self._get_pixell_mask()
            msk = reproject.healpix_from_enmap(self.pixell_mask,
                                               lmax=self.lmax,
                                               nside=self.nside)
            msk[msk < 0.95] = 0
        return msk

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            # 'Noise' contains the 2D Fourier space total noise power
            # spectrum from the ILC pipeline (this includes both
            # signal and instrument noise).
            pass
        return self.nl_coupled

    def _get_custom_beam(self, info):
        fname = info['file']
        beam_file = np.loadtxt(fname)
        ells = beam_file[0]
        beam = beam_file[1]
        pixwin = interp1d(ells, np.log(beam),
                          fill_value='extrapolate')
        ell = np.arange(3*self.nside)
        return np.exp(pixwin(ell))
