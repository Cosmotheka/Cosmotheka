import numpy as np
import healpy as hp
import pymaster as nmt
from scipy.interpolate import interp1d
from .utils import get_rerun_data, save_rerun_data


class MapperBase(object):
    """
    Base mapper class used as foundation \
    for the rest of mappers.
    """
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.beam_info = config.get('beam_info', [])
        self.nside = config['nside']
        self.nmt_field = None
        self.beam = None
        self.custom_auto = False
        # Option introduced to modify the Mode Coupling Matrix
        # In case the map has an implicit mask applied
        # See ACTk case for an example
        self.mask_power = config.get('mask_power', 1)
        self.coords = config['coords']
        self.mask = None

    def _get_rotator(self, coord_default):
        if self.coords != coord_default:
            rot = hp.Rotator(coord=[coord_default, self.coords])
        else:
            rot = None
        return rot

    def get_signal_map(self):
        """
        Returns the signal map of the mapper.

        Returns:
            delta_map (Array): signal mapper.
        """
        raise NotImplementedError("Do not use base class")

    def get_contaminants(self):
        """
        Returns the contaminants maps of the mapper.

        Returns:
            contaminants (Array): contaminant mapper.
        """
        raise NotImplementedError("Do not use base class")

    def get_mask(self):
        """
        Returns the mask of the mapper.

        Returns:
            mask (Array): mapper's mask
        """
        if self.mask is None:
            fn = '_'.join([f'mask_{self.mask_name}',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])
            self.mask = self._rerun_read_cycle(fn, 'FITSMap', self._get_mask)
        return self.mask

    def _get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        """
        Returns the coupled noise power spectrum of the mapper.

        Returns:
            nl_coupled (Array): coupled noise power spectrum
        """
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def _rerun_read_cycle(self, fname, ftype, func,
                          section=None, saved_by_func=False):
        d = get_rerun_data(self, fname, ftype,
                           section=section)
        if d is None:
            d = func()
            if not saved_by_func:
                save_rerun_data(self, fname, ftype, d)
        return d

    def _get_shifted_nz(self, dz, return_jk_error=False):
        z = self.dndz['z_mid']
        nz = self.dndz['nz']
        z_dz = z + dz
        sel = z_dz >= 0
        if return_jk_error:
            nz_jk = self.dndz['nz_jk']
            njk = len(nz_jk)
            enz = np.std(nz_jk, axis=0)*np.sqrt((njk-1)**2/njk)
            return np.array([z_dz[sel], nz[sel], enz[sel]])
        else:
            return np.array([z_dz[sel], nz[sel]])

    def get_ell(self):
        """
        Returns the array of multipoles associted with the \
        mapper's pixel resolution.

        Returns:
            ells (Array): multipoles array.
        """
        return np.arange(3 * self.nside)

    def _get_custom_beam(self, info):
        raise ValueError("This mapper does not support custom beams")

    def get_beam(self):
        """ Calculates the value of the mapper's beam at each \
            multipole. The beam is calculated following the \
            information contained in "self.beam_info". \
            Currently three types of beam are implemented:

                - Gaussian: a Gaussian beam defined by a FWHM \
                   in arcmin.
                - PixWin: the pixel window function associated \
                   resolution down/up-scalings.
                - Custom: loads beam from file.

            "self.beam_info" can contain information for many beams. \
            If this is the case, the final beam is the product of \
            individual beams.

            Returns:
                beam (Array): value of the beam at each multipole.
        """
        if self.beam is not None:
            return self.beam

        nside = self.nside
        self.beam = np.ones(3*nside)

        for info in self.beam_info:
            if info['type'] == 'Gaussian':
                ell = np.arange(3*nside)
                fwhm_amin = info['FWHM_arcmin']
                sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
                b = np.exp(-0.5 * ell * (ell + 1) * sigma_rad**2)
                b /= b[0]  # normalize it
                self.beam *= b
            elif info['type'] == 'PixWin':
                nside_native = info.get('nside_native', nside)
                ell_native = np.arange(3*nside_native)
                pixwin = interp1d(ell_native,
                                  np.log(hp.sphtfunc.pixwin(nside_native)),
                                  fill_value='extrapolate')
                ell = np.arange(3*nside)
                b = np.exp(pixwin(ell))
                self.beam *= b
            elif info['type'] == 'Custom':
                b = self._get_custom_beam(info)
                self.beam *= b
            else:
                raise NotImplementedError("Unknown beam type.")
        return self.beam

    def _get_nmt_field(self, signal=None, **kwargs):
        if signal is None:
            signal = self.get_signal_map(**kwargs)
        mask = self.get_mask(**kwargs)
        cont = self.get_contaminants(**kwargs)
        beam_eff = self.get_beam()

        n_iter = kwargs.get('n_iter', 0)
        return nmt.NmtField(mask, signal, beam=beam_eff,
                            templates=cont, n_iter=n_iter)

    def get_nmt_field(self, **kwargs):
        """
        Returns an instance of Namaster field given a mapper's \
        signal map, mask and beam.

        Returns:
            nmt_field (:class:`NaMaster.NmtField`): a Namaster \
            field instance. \
        """
        if self.nmt_field is None:
            self.nmt_field = self._get_nmt_field(signal=None, **kwargs)
        return self.nmt_field

    def get_dtype(self):
        """
        Returns the type of the mapper

        Returns:
            mapper_type (String)
        """
        raise NotImplementedError("Do not use base class")

    def get_spin(self):
        """
        Returns the spin of the mapper.

        Returns:
            spin (Int)
        """
        raise NotImplementedError("Do not use base class")
