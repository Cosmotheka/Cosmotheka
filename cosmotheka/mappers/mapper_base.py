# fmt: off
import numpy as np
import healpy as hp
import pymaster as nmt
from scipy.interpolate import interp1d
from .utils import get_rerun_data, save_rerun_data, rotate_mask


class MapperBase(object):
    """
    Base mapper class used as foundation \
    for the rest of mappers.
    """
    # map_name is the name that will be used for rerun signal map files.
    map_name = None

    masked_on_input = False  # If True, the mapper's signal map is masked

    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        """
        Reads the configuration file to obtain \
        the desired resolution and coordinates.
        Initialises the NaMaster field, signal \
        map, mask and beam as None.
        """
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.beam_info = config.get('beam_info', [])
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.nmt_field = None
        self.beam = None
        self.custom_auto = False
        # Option introduced to modify the Mode Coupling Matrix
        # In case the map has an implicit mask applied
        # See ACTk case for an example
        self.mask_power = config.get('mask_power', 1)
        self.coords = config['coords']
        self.mask = None
        self.signal_map = None
        # dndz needs to be defined for _get_shifted_nz. We should consider
        # creating a subclass for Nz tracers, as in CCL.
        self.dndz = None
        # Remove overlap? You should pass a dictionary with name & mask
        self.remove_overlap = config.get("remove_overlap")
        if self.remove_overlap is not None:
            self.map_name += '_removed_overlap'
            for k in self.remove_overlap.keys():
                self.map_name += f'_{k}'
        self.rot = None  # Defined in the child classes

    def _get_rotator(self, coord_default):
        if self.coords != coord_default:
            rot = hp.Rotator(coord=[coord_default, self.coords])
        else:
            rot = None
        return rot

    def _get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_signal_map(self, **kwargs):
        """
        If the mapper.signal_map is not None, returns it.
        Otherwise, asks _get_signal_map() to compute \
        it and assings it to mapper.signal_map.

        Returns:
            signal map (Array or None): mapper's signal map
        """
        if self.signal_map is None:
            fn = '_'.join([f'{self.map_name}_signal_map',
                           *[f'{k}{v}' for k, v in kwargs.items()],
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])

            def func():
                if not kwargs:
                    return self._get_signal_map()
                else:
                    return self._get_signal_map(**kwargs)

            signal_map = self._rerun_read_cycle(fn, 'FITSMap', func)
            self.signal_map = np.atleast_2d(signal_map)

        return self.signal_map

    def get_contaminants(self):
        """
        Returns the contaminants maps of the mapper.

        Returns:
            contaminants (Array or None): contaminant map.
        """
        return None

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

            def get_final_mask():
                # Removing the overlapping area at the mask level. Note that
                # this is an approximation and weights, biases, etc might need
                # to be recomputed.
                if self.remove_overlap is not None:
                    msk = self._get_mask()
                    for v in self.remove_overlap.values():
                        m = hp.read_map(v)
                        m[m == hp.UNSEEN] = 0
                        m = rotate_mask(m, self.rot)
                        m = hp.ud_grade(m, nside_out=self.nside)
                        # Set filled pixels to 0
                        msk[m != 0] = 0
                else:
                    msk = self._get_mask()

                return msk

            self.mask = self._rerun_read_cycle(fn, 'FITSMap', get_final_mask)
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
                          section=None, saved_by_func=False, **func_kwargs):
        print(f"Rerun read cycle for {fname} of type {ftype}", flush=True)
        d = get_rerun_data(self, fname, ftype,
                           section=section)
        print(f"Data loaded: {d is not None}", flush=True)
        if d is None:
            d = func(**func_kwargs)
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
                            templates=cont, n_iter=n_iter,
                            masked_on_input=self.masked_on_input)

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
