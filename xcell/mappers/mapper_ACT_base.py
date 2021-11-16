from .mapper_base import MapperBase
from pixell import enmap, reproject
import numpy as np
import pymaster as nmt


class MapperACTBase(MapperBase):
    def __init__(self, config):
        self._get_ACT_defaults(config)

    def _get_ACT_defaults(self, config):
        self._get_defaults(config)
        self.file_map = config['file_map']
        self.file_mask = config['file_mask']
        self.file_noise = config.get('file_noise', None)
        self.file_cross_noise = config.get('file_cross_noise', None)
        self.file_weights = config.get('file_weights', None)
        self.lmax = config.get('lmax', 6000)
        self.signal_map = None
        self.mask = None
        self.pixell_mask = None
        self.noise = None
        self.cross_noise = None
        self.nl_coupled = None
        self.weights = None
        self.beam_info = None
        self.beam = None

    def get_signal_map(self):
        return NotImplementedError("Do not use base class")

    def _get_pixell_mask(self):
        if self.pixell_mask is None:
            self.pixell_mask = enmap.read_map(self.file_mask)
        return self.pixell_mask

    def get_mask(self):
        if self.mask is None:
            self.pixell_mask = self._get_pixell_mask()
            self.mask = reproject.healpix_from_enmap(self.pixell_mask,
                                                     lmax=self.lmax,
                                                     nside=self.nside)
        return self.mask

    def get_noise(self):
        if self.noise is None:
            self.noise = enmap.read_map(self.file_noise)
            self.noise = reproject.healpix_from_enmap(self.noise,
                                                      lmax=self.lmax,
                                                      nside=self.nside)
        return self.noise

    def get_cross_noise(self):
        if self.cross_noise is None:
            self.cross_noise = enmap.read_map(self.file_cross_noise)
            self.cross_noise = reproject.healpix_from_enmap(self.cross_noise,
                                                            lmax=self.lmax,
                                                            nside=self.nside)
        return self.cross_noise

    def get_weights(self):
        if self.weights is None:
            self.weights = enmap.read_map(self.file_weights)
            self.weights = reproject.healpix_from_enmap(self.weights,
                                                        lmax=self.lmax,
                                                        nside=self.nside)
        return self.weights

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            noise = self.get_noise()
            noise_f = self._get_nmt_field(signal=noise)
            self.nl_coupled = nmt.compute_coupled_cell(noise_f, noise_f)
        return self.nl_coupled

    def _beam_gaussian(self, ell, fwhm_amin):
        sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
        return np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)

    def get_beam(self):
        if self.beam is None:
            if self.beam_info is None:  # No beam
                self.beam = np.ones(3*self.nside)
            else:
                ell = np.arange(3*self.nside)
                self.beam = self._beam_gaussian(ell, self.beam_info)
                self.beam /= self.beam[0]
        return self.beam
