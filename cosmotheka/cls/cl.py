#!/usr/bin/python
from .data import Data
from .theory import Theory
from . import tools
import numpy as np
import pymaster as nmt
import os
import warnings


class ClBase():
    """
    ClBase class. It contains the basic and common methods for data and
    fiducial Cell handling and computation.
    """
    def __init__(self, data, tr1, tr2, ignore_existing_yml=False):
        """
        Parameters
        ----------
        data: dict
            Configuration dictionary (e.g. read yaml)
        tr1: str
            First tracer
        tr2: str
            Second tracer
        ignore_existing_yml: bool
            If True, ignore existing yaml in the output directory and use the
            input configuration. Otherwise, use the existing yaml.
        """
        self.data = Data(data=data, ignore_existing_yml=ignore_existing_yml)
        self.tr1 = tr1
        self.tr2 = tr2
        self.nside = self.data.data['sphere']['nside']
        self._read_symmetric = self.data.read_symmetric(tr1, tr2)
        if self._read_symmetric:
            warnings.warn('Reading/computing the symmetric element.')
        self._mapper1 = None
        self._mapper2 = None
        #
        self.cl_file = None
        self.ell = None
        self.cl = None
        self.cl_cp = None

    def get_outdir(self, subdir=''):
        """
        Return the output directory.

        Parameters
        ----------
        subdir: str
            Subfolder name inside the output directory where to save the output
            files.

        Return
        ------
            Path to the output directory or subdirectory.
        """
        root = self.data.data['output']
        if self._read_symmetric:
            trreq = self.data.get_tracers_bare_name_pair(self.tr2, self.tr1,
                                                         '_')
        else:
            trreq = self.data.get_tracers_bare_name_pair(self.tr1, self.tr2,
                                                         '_')
        outdir = os.path.join(root, subdir, trreq)
        return outdir

    def get_mappers(self):
        """
        Return the mappers of the correlated tracers

        Return
        ------
        mapper1: cosmotheka.mappers.XXX
            Mapper of the first tracer
        mapper2: cosmotheka.mappers.XXX
            Mapper of the second tracer
        """
        if self._mapper1 is None:
            self._mapper1 = self.data.get_mapper(self.tr1)
            self._mapper2 = self._mapper1 if self.tr1 == self.tr2 else \
                self.data.get_mapper(self.tr2)
        return self._mapper1, self._mapper2

    def get_cl_file(self):
        """
        Return the dictionary with the computed Cell, ells, etc. The specific
        output will depend on the child class implementation.
        """
        raise ValueError('Cl_Base class is not to be used directly!')

    def get_ell_cl(self):
        """
        Return the noiseless computed ell and Cell.

        Return
        ------
        ell: numpy.array
            The angular modes. In the case of a data Cell, these will be the
            bandpowers (i.e. bins on ell).

        Cell: numpy.array
            Noiseless angular power spectrum with shape (ncls, nell), where
            ncls is the number of possible cls and nell the number of
            multipoles. In the case of a data Cell, these will be the
            bandpowers (i.e. bins on ell).
        """
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.cl

    def get_ell_cl_cp(self):
        """
        Return the noiseless coupled cell

        Return
        ------
        ell: numpy.array
            The angular modes up to lmax = 3*nside - 1
        Cell: numpy.array
            Noiseless coupled angular power spectrum with shape (ncls,
            3*nside), where ncls is the number of possible Cells.
        """
        if self.ell is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.cl_cp

    def get_n_cls(self):
        """
        Return the number of possible Cells

        Return
        ------
        ncls: float
            Number of possible Cells
        """
        s1, s2 = self.get_spins()
        nmap1 = 1 + (s1 > 0)
        nmap2 = 1 + (s2 > 0)
        return nmap1 * nmap2

    def get_spins(self):
        """
        Return the tracers' spin.

        Return
        ------
        s1: float
            Spin of tracer 1
        s2: float
            Spin of tracer 2
        """
        mapper1, mapper2 = self.get_mappers()
        s1 = mapper1.get_spin()
        s2 = mapper2.get_spin()
        return s1, s2

    def get_dtypes(self):
        """
        Return the tracers' data types (e.g. galaxy_density).

        Return
        ------
        d1: str
            Data type of tracer 1
        d2: str
            Data type of tracer 2
        """
        mapper1, mapper2 = self.get_mappers()
        d1 = mapper1.get_dtype()
        d2 = mapper2.get_dtype()
        return d1, d2


class Cl(ClBase):
    """
    Cl class. This is the class used to compute the data angular power spectrum
    with a pseudo-Cell estimator (as implemented in NaMaster).
    """
    def __init__(self, data, tr1, tr2, ignore_existing_yml=False):
        """
        Parameters
        ----------
        data: dict
            Configuration dictionary (e.g. read yaml)
        tr1: str
            First tracer
        tr2: str
            Second tracer
        ignore_existing_yml: bool
            If True, ignore existing yaml in the output directory and use the
            input configuration. Otherwise, use the existing yaml.
        """
        super().__init__(data, tr1, tr2, ignore_existing_yml)
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.b = self.get_NmtBin()
        self.recompute_cls = self.data.data['recompute']['cls']
        self.recompute_mcm = self.data.data['recompute']['mcm']
        # Not needed to load cl if already computed
        self._w = None
        self._wcov = None
        ##################
        self.nl = None
        self.nl_cp = None
        self.cl_cp = None
        self.wins = None
        self.cls_cov = None

    def get_NmtBin(self):
        """
        Return the pymaster.NmtBin instance with the requeseted binning.

        Return
        ------
        b: pymaster.NmtBin
            Binning
        """
        if self._read_symmetric:
            trs = self.data.get_tracers_bare_name_pair(self.tr2, self.tr1)
        else:
            trs = self.data.get_tracers_bare_name_pair(self.tr1, self.tr2)
        if ((trs in self.data.data['cls']) and
                ('bpw_edges' in self.data.data['cls'][trs].keys())):
            bpw_edges = np.array(self.data.data['cls'][trs]['bpw_edges'])
        else:
            bpw_edges = np.array(self.data.data['bpw_edges'])
        nside = self.nside
        # 3*nside == ells[-1] + 1
        bpw_edges = bpw_edges[bpw_edges <= 3 * nside]
        # Exhaust lmax --> gives same result as previous method,
        # but adds 1 bpw (not for 4096)
        if 3*nside not in bpw_edges:
            bpw_edges = np.append(bpw_edges, 3*nside)
        b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
        return b
    
    # New code added for catalogue implementation
    # Previous function did not have use_maps argument
    def get_nmt_fields(self, use_maps=False):
        """
        Return the pymaster.NmtField instances of the correlated tracers.

        Return
        ------
        f1: pymaster.NmtField
            Field of tracer 1
        f2: pymaster.NmtField
            Field of tracer 2
        """
        mapper1, mapper2 = self.get_mappers()
        if use_maps:
            f1 = mapper1.get_nmt_field(for_cov=True)
            f2 = mapper2.get_nmt_field(for_cov=True)
        else:
            f1 = mapper1.get_nmt_field()
            f2 = mapper2.get_nmt_field()
        return f1, f2

    def get_workspace(self, read_unbinned_MCM=True):
        """
        Return the pymaster.NmtWorkspace instance with the mode-coupling matrix
        of the correlated fields.

        Parameters
        ----------
        read_unbinned_MCM: bool
            If True, load the unbinned mode-coupling matrix as well

        Return
        ------
        w: pymaster.NmtWorkspace
            Workspace with the mode-coupling matrix of both tracers
        """
        if self._w is None:
            self._w = self._compute_workspace(read_unbinned_MCM=read_unbinned_MCM, use_maps=False)
        return self._w

    def get_workspace_cov(self):
        """
        Return the pymaster.NmtWorkspace instance with the mode-coupling matrix
        of the correlated fields used to compute the covariance. It can be
        different from the one to compute the Cells if requested in the
        configuration file (e.g. for the spin-0 approximation).

        Return
        ------
        w: pymaster.NmtWorkspace
            Workspace with the mode-coupling matrix of both tracers used in the
            covariance calculation.
        """
        if self._wcov is None:
            spin0 = self.data.data['cov'].get('spin0', False)
            if spin0 and (self.get_spins() != (0, 0)):
                self._wcov = self._compute_workspace(spin0=spin0,
                                                     read_unbinned_MCM=False, 
                                                     use_maps=True)
            else:
                self._wcov = self._compute_workspace(read_unbinned_MCM=False, use_maps=True)

        return self._wcov

    def _compute_workspace(self, spin0=False, read_unbinned_MCM=True, use_maps=False):
        """
        Return the pymaster.NmtWorkspace with the mode-coupling matrix of the
        correlated fields.

        Parameters
        ----------
        spin0: bool
            If True, compute the workspace assuming th all fields have
            spin-0.
        read_unbinned_MCM: bool
            If True, load the unbinned mode-coupling matrix as well

        Return
        ------
        w: pymaster.NmtWorkspace
            Workspace with the mode-coupling matrix of both tracers used.
        """
        # Check if the fields are already of spin0 to avoid computing the
        # workspace twice
        spin0 = spin0 and (self.get_spins() != (0, 0))
        mask1, mask2 = self.get_masks_names()
        if self._read_symmetric:
            mask1, mask2 = mask2, mask1
        
        #if spin0:
        #    fname = os.path.join(self.outdir, f'w0__{mask1}__{mask2}.fits')
        #else:
        #    fname = os.path.join(self.outdir, f'w__{mask1}__{mask2}.fits')

        # These three lines replace the commented out if/else statement above
        tag = "map" if use_maps else "data"
        spin_tag = "w0" if spin0 else "w"
        fname = os.path.join(self.outdir, f'{spin_tag}_{tag}__{mask1}__{mask2}.fits')

        w = nmt.NmtWorkspace()
        if (not self.recompute_mcm) and os.path.isfile(fname):
            tools.read_wsp(w, fname, read_unbinned_MCM=read_unbinned_MCM)
            if w.wsp is not None:
                return w

        l_toeplitz, l_exact, dl_band = self.data.check_toeplitz('cls')
        if spin0:
            m1, m2 = self.get_mappers()
            msk1 = m1.get_mask()
            msk2 = m2.get_mask()
            f1 = nmt.NmtField(msk1, None, spin=0)
            f2 = nmt.NmtField(msk2, None, spin=0)
        else:
            f1, f2 = self.get_nmt_fields(use_maps=use_maps)
        if use_maps:
            assert f1.maps is not None and f2.maps is not None, \
                "Covariance workspace must be built from map-based NmtFields"

        w.compute_coupling_matrix(f1, f2, self.b,
                                  l_toeplitz=l_toeplitz, l_exact=l_exact,
                                  dl_band=dl_band)
        tools.save_wsp(w, fname)
        self.recompute_mcm = False

        return w

    def _get_cl_crude_error(self, cl_cp, mean_mamb):
        b = self.b
        nbpw = b.get_n_bands()

        # To avoid dividing by 0, set mean_mamb to a small number
        if mean_mamb == 0:
            mean_mamb = 1e-10

        cl = b.bin_cell(cl_cp)
        err = np.zeros_like(cl)
        for i in range(nbpw):
            ells_in_i = b.get_ell_list(i)

            # Reshape to have shape (ncls, nells)
            w_i = b.get_weight_list(i)[None, :]
            cli = cl[:, i][:, None]
            cl_cpi = cl_cp[:, ells_in_i]
            # Number of non-zero weighs. We correct by M-1/M to take into
            # account we have just a few elements per bin
            nw_i = np.sum(w_i != 0)
            sigma2 = np.sum(w_i*(cl_cpi - cli)**2, axis=1)
            sigma2 /= ((nw_i - 1)/nw_i*np.sum(w_i))
            err[:, i] = np.sqrt(sigma2)

        return err / mean_mamb

    def get_ell_cl_crude_error(self):
        """
        Return a crude estimation of the cl errors.

        Return
        ------
        ell: numpy.array
            The bandpowers.

        crude_err: numpy.array
            The estimated errors
        """
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.crude_err

    def get_cl_file(self):
        """
        Return the dictionary with the computed Cell, ells, etc.

        Return
        ------
        cl_file: dict
            Dictionary with the computed quantities. The keys are
             - ell: bandpowers,
             - cl: noiseless uncoupled binned power spectrum,
             - cl_cp: noiseless coupled power spectrum,
             - nl: noise uncoupled binned power spectrum,
             - nl_cp: noise coupled power spectrum,
             - cl_cov_cp: the C_ell that should be used for the
               auto-correlation of this mapper when calculating any
               cross-covariance involving it. E.g. if we call this field "a",
               this will play the role of C_ell^aa when computing the
               cross-covariance Cov(C_ell^ab, C_ell^aa) or
               Cov(C_ell^ab,C_ell^ac).
             - cl_cov_11_cp: when computing the auto-covariance of this
               mapper's auto-correlation, it will be computed as the
               auto-covariance of a general power spectrum C_ell^12, which
               involves 3 power spectra: C_ell^11, C_ell^12, C_ell^22.
             - cl_cov_12_cp: as above for c_ell^12,
             - cl_cov_22_cp: as above for c_ell^22,
             - wins: bandpower window functions,
             - correction: a mapper level correction applied to the power
               spectra
        """
        if self._read_symmetric:
            fname = os.path.join(self.outdir, f'cl_{self.tr2}_{self.tr1}.npz')
        else:
            fname = os.path.join(self.outdir, f'cl_{self.tr1}_{self.tr2}.npz')
        ell = self.b.get_effective_ells()
        recompute = self.recompute_cls or self.recompute_mcm
        if recompute or (not os.path.isfile(fname)):
            print(f"Computing Cell for {self.tr1} {self.tr2}")
            mapper1, mapper2 = self.get_mappers()
            f1, f2 = self.get_nmt_fields()
            f1c, f2c = self.get_nmt_fields(use_maps=True)
            # f1, f2   : data / catalogue fields (for Cl estimation)
            # f1c, f2c : map-based fields (for covariance inputs)   
            w = self.get_workspace()
            wins = w.get_bandpower_windows()
            w_a = mapper1.get_mask()
            w_b = mapper2.get_mask()
            mean_mamb = np.mean(w_a * w_b)

            # Compute power spectrum
            # If auto-correlation, compute noise and,
            # if needed, the custom signal power spectrum.
            auto = self.tr1 == self.tr2
            # Noise
            if auto:
                nl_cp = mapper1.get_nl_coupled()
                nl = w.decouple_cell(nl_cp)
            else:
                n_cls = self.get_n_cls()
                nl_cp = np.zeros((n_cls, 3 * self.nside))
                nl = np.zeros([n_cls, self.b.get_n_bands()])
            # Signal
            if auto and mapper1.custom_auto:
                # **Note** that no mappers currently exist where this
                # custom auto-correlations are needed and which are
                # also catalog-based, so we are implicitly assuming
                # here that all mappers are map-based.
                cl_cp = mapper1.get_cl_coupled()
                cl = w.decouple_cell(cl_cp)
                cls_cov = mapper1.get_cls_covar_coupled()
                # This function should return a dictionary with the
                # following contents:
                #  - 'cross': the C_ell that should be used for the
                #       auto-correlation of this mapper when
                #       calculating any cross-covariance involving
                #       it. E.g. if we call this field "a", this will
                #       play the role of C_ell^aa when computing the
                #       cross-covariance Cov(C_ell^ab, C_ell^aa)
                #       or Cov(C_ell^ab,C_ell^ac).
                #  - 'auto_11/12/22': when computing the
                #       auto-covariance of this mapper's auto-correlation,
                #       it will be computed as the auto-covariance of
                #       a general power spectrum C_ell^12, which involves
                #       3 power spectra: C_ell^11, C_ell^12, C_ell^22.
                #       These are given by the corresponding entries in
                #       the dictionary.
                cl_cov_cp = cls_cov['cross']
                cl_cov_11_cp = cls_cov['auto_11']
                cl_cov_12_cp = cls_cov['auto_12']
                cl_cov_22_cp = cls_cov['auto_22']
                # mapper.get_cl_coupled is assumed to return
                # the noise-less power spectrum. No need
                # to subtract it here.
            else:
                cl_cov_cp = nmt.compute_coupled_cell(f1c, f2c)
                # A standard auto-correlation auto-covariance
                # is just ~propto 2*C_ell^2 rather than
                # (C_ell^11 C_ell^22+C_ell^12^2), which can be
                # achieved by equating all 3 C_ells.
                cl_cov_11_cp = cl_cov_cp
                cl_cov_12_cp = cl_cov_cp
                cl_cov_22_cp = cl_cov_cp
                if (f1 == f1c) and (f2 == f2c):
                    # With the check above, this means that both
                    # fields are map-based.
                    cl = w.decouple_cell(cl_cov_cp)
                    cl_cp = cl_cov_cp - nl_cp
                    cl -= nl
                else:
                    cl_cp = nmt.compute_coupled_cell(f1, f2)
                    cl = w.decouple_cell(cl_cp)

            # Note that while we have subtracted the noise
            # bias from `cl_cp`, `cl_cov_cp` still includes it.
            correction = 1
            if (mean_mamb != 0) and ((mapper1.mask_power > 1) or
                                     (mapper2.mask_power > 1)):
                # Applies correction factor if masks have been
                # implicitly applied to the maps
                # See ACTk for reference
                n_a = mapper1.mask_power
                n_b = mapper2.mask_power

                correction = mean_mamb/np.mean(w_a**n_a*w_b**n_b)
                print("correction", correction)
                # Apply correction to all Cl's
                cl *= correction
                cl_cp *= correction
                cl_cov_cp *= correction
                cl_cov_11_cp *= correction
                cl_cov_12_cp *= correction
                cl_cov_22_cp *= correction

            # Crude estimation of the error
            crude_err = self._get_cl_crude_error(cl_cov_cp, mean_mamb)

            tools.save_npz(fname, ell=ell, cl=cl, cl_cp=cl_cp, nl=nl,
                           nl_cp=nl_cp, cl_cov_cp=cl_cov_cp,
                           cl_cov_11_cp=cl_cov_11_cp,
                           cl_cov_12_cp=cl_cov_12_cp,
                           cl_cov_22_cp=cl_cov_22_cp, wins=wins,
                           correction=correction, mean_mamb=mean_mamb,
                           crude_err=crude_err)
            self.recompute_cls = False

        cl_file = np.load(fname)
        cl = cl_file['cl']
        if np.any(ell != cl_file['ell']):
            raise ValueError('The file {} does not have the expected bpw. \
                             Aborting!'.format(fname))

        self.cl_file = cl_file
        self.ell = cl_file['ell']
        self.cl = cl_file['cl']
        self.cl_cp = cl_file['cl_cp']
        self.nl = cl_file['nl']
        self.nl_cp = cl_file['nl_cp']
        self.wins = cl_file['wins']
        self.cls_cov = {'cross': cl_file['cl_cov_cp'],
                        'auto_11': cl_file['cl_cov_11_cp'],
                        'auto_12': cl_file['cl_cov_12_cp'],
                        'auto_22': cl_file['cl_cov_22_cp']}
        self.mean_mamb = cl_file['mean_mamb']
        self.crude_err = cl_file['crude_err']

        return cl_file

    def get_ell_nl(self):
        """
        Return the ell and uncopuled binned noise power spectrum.

        Return
        ------
        ell: numpy.array
            The bandpowers.
        nl: numpy.array
            Uncopuled binned noise power spectrum.
        """
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.nl

    def get_ell_nl_cp(self):
        """
        Return the ell and copuled noise power spectrum.

        Return
        ------
        ell: numpy.array
            The angular modes (up to lmax = 3nside-1).

        nl: numpy.array
            Uncopuled binned noise power spectrum.
        """
        if self.nl_cp is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.nl_cp

    def get_ell_cl_cp_cov(self):
        """
        Return the noiseless coupled cell to be used for the covariance. This
        correspond to the C_ell that should be used for the auto-correlation of
        this mapper when calculating any cross-covariance involving it. E.g. if
        we call this field "a", this will play the role of C_ell^aa when
        computing the cross-covariance Cov(C_ell^ab, C_ell^aa) or
        Cov(C_ell^ab,C_ell^ac).

        Return
        ------
        ell: numpy.array
            The angular modes up to lmax = 3*nside - 1
        Cell: numpy.array
            Noiseless coupled angular power spectrum with shape (ncls,
            3*nside), where ncls is the number of possible Cells.
        """
        if self.ell is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.cls_cov['cross']

    def get_ell_cls_cp_cov_auto(self):
        """
        Return the noiseless coupled cell to be used for the covariance. This
        correspond to the C_ell that should be used when computing the
        auto-covariance of this mapper's auto-correlation, it will be computed
        as the auto-covariance of a general power spectrum C_ell^12, which
        involves 3 power spectra: C_ell^11, C_ell^12, C_ell^22.

        Return
        ------
        ell: numpy.array
            The angular modes up to lmax = 3*nside - 1
        cl11: numpy.array
            Noiseless coupled angular power spectrum with shape (ncls,
            3*nside), where ncls is the number of possible Cells. It
            corresponds to C_ell^11
        cl12: numpy.array
            Noiseless coupled angular power spectrum with shape (ncls,
            3*nside), where ncls is the number of possible Cells. It
            corresponds to C_ell^12
        cl22: numpy.array
            Noiseless coupled angular power spectrum with shape (ncls,
            3*nside), where ncls is the number of possible Cells. It
            corresponds to C_ell^22
        """
        if self.ell is None:
            self.get_cl_file()
        cl11 = self.cls_cov['auto_11']
        cl12 = self.cls_cov['auto_12']
        cl22 = self.cls_cov['auto_22']
        ell = np.arange(3 * self.nside)
        return ell, cl11, cl12, cl22

    def get_masks(self):
        """
        Return the tracers' masks.

        Return
        ------
        m1: numpy.array
            HEALPix map corresponding to the first tracer mask
        m2: numpy.array
            HEALPix map corresponding to the second tracer mask
        """
        mapper1, mapper2 = self.get_mappers()
        m1 = mapper1.get_mask()
        m2 = mapper2.get_mask()
        return m1, m2

    def get_mean_mamb(self):
        """
        Return the mean of the tracers' masks product.

        Return
        ------
        <m1*m2>: float
            Mean of the product of the tracers' masks
        """
        if self.ell is None:
            self.get_cl_file()
        return self.mean_mamb

    def get_masks_names(self):
        """
        Return the tracers' mask names.

        Return
        ------
        m1: str
            Name of the first tracer's mask
        m2: str
            Name of the second tracer's mask
        """
        mapper1, mapper2 = self.get_mappers()
        m1 = mapper1.mask_name
        m2 = mapper2.mask_name
        return m1, m2

    def get_bandpower_windows(self):
        """
        Return the bandpower windows. Applying these to a fiducial Cell
        corresponds to the coupling-binning-uncoupling operation.

        Return
        ------
        wins: numpy.array
            Bandpower window function
        """
        if self.ell is None:
            self.get_cl_file()
        return self.wins


class ClFid(ClBase):
    """
    ClFid class. This is the class used to compute the fiducial angular power
    spectrum from theory.
    """
    def __init__(self, data, tr1, tr2, ignore_existing_yml=False):
        """
        Parameters
        ----------
        data: dict
            Configuration dictionary (e.g. read yaml)
        tr1: str
            First tracer
        tr2: str
            Second tracer
        ignore_existing_yml: bool
            If True, ignore existing yaml in the output directory and use the
            input configuration. Otherwise, use the existing yaml.
        """
        super().__init__(data, tr1, tr2, ignore_existing_yml)
        self.supported_dtypes = ['galaxy_density',
                                 'galaxy_shear',
                                 'cmb_tSZ',
                                 'cmb_convergence']
        m1, m2 = self.get_mappers()
        dt1 = m1.get_dtype()
        dt2 = m2.get_dtype()
        if ((dt1 not in self.supported_dtypes) or
                (dt2 not in self.supported_dtypes)):
            raise NotImplementedError("Fiducial C_ells cannot be "
                                      f"computed for types {dt1}, {dt2}")

        self.outdir = self.get_outdir('fiducial')
        os.makedirs(self.outdir, exist_ok=True)
        self.th = Theory(self.data.data)
        self._ccl_tr1 = None
        self._ccl_tr2 = None
        self.cl_data = Cl(data, tr1, tr2, ignore_existing_yml=True)
        self.cl_binned = None  # This is the one you compare with data
        self.ell_binned = None

    def get_tracers_ccl(self):
        """
        Return ccl tracer and extra information

        Return
        ------
        ccl_tr1: dict
            A dictionary for tracer1 with keys:
             - 'name': the input tracer name
             - 'ccl_tr': Instance of ccl.tracers.Tracer
             - 'ccl_pr': Instance of ccl.halos.profiles
             - 'ccl_pr_2pt': Instance of ccl.halos.profiles_2pt
             - 'with_hm': True if halo model is used (i.e. if 'use_halo_model'
               in tracer config
             - 'normed': True if the profiles are normalized
        ccl_tr2: dict
            A dictionary as ccl_tr1 but for the second tracer.

        """
        if self._ccl_tr1 is None:
            mapper1, mapper2 = self.get_mappers()
            trlist = self.data.data['tracers']
            self._ccl_tr1 = self.th.compute_tracer_ccl(self.tr1,
                                                       trlist[self.tr1],
                                                       mapper1)
            self._ccl_tr2 = self.th.compute_tracer_ccl(self.tr2,
                                                       trlist[self.tr2],
                                                       mapper2)
        return self._ccl_tr1, self._ccl_tr2

    def get_cl_file(self):
        """
        Return the dictionary with the computed Cell, ells, etc.

        Return
        ------
        cl_file: dict
            Dictionary with the computed quantities. The keys are
             - ell: multipoles up to lmax=3*nside -1,
             - cl: fiducial power spectrum,
             - cl_cp: fiducial coupled power spectrum
             - ell_binned: bandpowers
             - cl_binned: coupled-binned-uncoupled fiducial power spectrum
        """
        nside = self.data.data['sphere']['nside']
        if self._read_symmetric:
            fname = os.path.join(self.outdir, f'cl_{self.tr2}_{self.tr1}.npz')
        else:
            fname = os.path.join(self.outdir, f'cl_{self.tr1}_{self.tr2}.npz')
        ell = np.arange(3 * nside)
        if not os.path.isfile(fname):
            print(f"Computing fiducial Cell for {self.tr1} {self.tr2}")
            ccl_tr1, ccl_tr2 = self.get_tracers_ccl()
            cl = self.th.get_ccl_cl(ccl_tr1, ccl_tr2, ell)
            b1 = self.data.get_bias(self.tr1)
            b2 = self.data.get_bias(self.tr2)
            cl *= b1*b2
            tracers = self.data.data['tracers']
            d1, d2 = self.get_dtypes()
            for tr, dtype in zip([self.tr1, self.tr2], [d1, d2]):
                if (dtype == 'galaxy_shear'):
                    m = tracers[tr].get('m', 0)
                    cl = (1 + m) * cl

            # This is only valid for LCDM and spins 0 and 2.
            s1, s2 = self.get_spins()
            size = s1 + s2
            if size == 0:
                size = 1
            cl_vector = np.zeros((size, cl.size))
            cl_vector[0] = cl

            w = self.cl_data.get_workspace()
            cl_cp = w.couple_cell(cl_vector)
            cl_binned = w.decouple_cell(cl_cp)
            ell_binned = self.cl_data.get_ell_cl()[0]
            tools.save_npz(fname, cl=cl_vector, ell=ell, cl_cp=cl_cp,
                           ell_binned=ell_binned, cl_binned=cl_binned)

        cl_file = np.load(fname)
        if np.any(cl_file['ell'] != ell):
            raise ValueError(f'The ell in {fname} does not match the ell \
                               from nside={nside}')

        self.ell = cl_file['ell']
        self.cl = cl_file['cl']
        self.cl_cp = cl_file['cl_cp']
        self.cl_binned = cl_file['cl_binned']
        self.ell_binned = cl_file['ell_binned']
        return cl_file

    def get_ell_cl_binned(self):
        """
        Return the binned Cell (i.e. decouple(couple(cl))). This one is the one
        you compare with data.

        Return
        ------
        ell_binned: numpy.array
            Bandpowers

        cl_binned: numpy.array
            Coupled-binned-uncoupled fiducial power spectrum
        """
        if self.ell is None:
            self.get_cl_file()
        return self.ell_binned, self.cl_binned


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from \
                                     data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('tr1', type=str, help='Tracer 1 name')
    parser.add_argument('tr2', type=str, help='Tracer 2 name')
    parser.add_argument('--fiducial', default=False, action='store_true',
                        help='Compute the fiducial model Cl')
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    if args.fiducial:
        cl = ClFid(data, args.tr1, args.tr2)
    else:
        cl = Cl(data, args.tr1, args.tr2)
    cl.get_cl_file()
