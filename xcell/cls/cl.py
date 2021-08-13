#!/usr/bin/python
from .data import Data
import numpy as np
import pyccl as ccl
import pymaster as nmt
import os
import warnings


class ClBase():
    def __init__(self, data, tr1, tr2):
        self.data = Data(data=data)
        self.tr1 = tr1
        self.tr2 = tr2
        self._read_symmetric = self.data.read_symmetric(tr1, tr2)
        if self._read_symmetric:
            warnings.warn('Reading/computing the symmetric element.')
        self._mapper1 = None
        self._mapper2 = None
        #
        self.cl_file = None
        self.ell = None
        self.cl = None

    def get_outdir(self, subdir=''):
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
        if self._mapper1 is None:
            self._mapper1 = self.data.get_mapper(self.tr1)
            self._mapper2 = self._mapper1 if self.tr1 == self.tr2 else \
                self.data.get_mapper(self.tr2)
        return self._mapper1, self._mapper2

    def get_cl_file(self):
        raise ValueError('Cl_Base class is not to be used directly!')

    def get_ell_cl(self):
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.cl

    def get_n_cls(self):
        s1, s2 = self.get_spins()
        nmap1 = 1 + (s1 > 0)
        nmap2 = 1 + (s2 > 0)
        return nmap1 * nmap2

    def get_spins(self):
        mapper1, mapper2 = self.get_mappers()
        s1 = mapper1.get_spin()
        s2 = mapper2.get_spin()
        return s1, s2

    def get_dtypes(self):
        mapper1, mapper2 = self.get_mappers()
        d1 = mapper1.get_dtype()
        d2 = mapper2.get_dtype()
        return d1, d2


class Cl(ClBase):
    def __init__(self, data, tr1, tr2):
        super().__init__(data, tr1, tr2)
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.nside = self.data.data['healpy']['nside']
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
        if self._read_symmetric:
            trs = self.data.get_tracers_bare_name_pair(self.tr2, self.tr1)
        else:
            trs = self.data.get_tracers_bare_name_pair(self.tr1, self.tr2)
        if 'bpw_edges' in self.data.data['cls'][trs].keys():
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

    def get_nmt_fields(self):
        mapper1, mapper2 = self.get_mappers()
        f1 = mapper1.get_nmt_field()
        f2 = mapper2.get_nmt_field()
        return f1, f2

    def get_workspace(self):
        if self._w is None:
            self._w = self._compute_workspace()
        return self._w

    def get_workspace_cov(self):
        if self._wcov is None:
            spin0 = self.data.data['cov'].get('spin0', False)
            if spin0 and (self.get_spins() != (0, 0)):
                self._wcov = self._compute_workspace(spin0=spin0)
            else:
                self._wcov = self.get_workspace()

        return self._wcov

    def _compute_workspace(self, spin0=False):
        # Check if the fields are already of spin0 to avoid computing the
        # workspace twice
        spin0 = spin0 and (self.get_spins() != (0, 0))
        mask1, mask2 = self.get_masks_names()
        if self._read_symmetric:
            mask1, mask2 = mask2, mask1
        if spin0:
            fname = os.path.join(self.outdir, f'w0__{mask1}__{mask2}.fits')
        else:
            fname = os.path.join(self.outdir, f'w__{mask1}__{mask2}.fits')
        w = nmt.NmtWorkspace()
        if self.recompute_mcm or (not os.path.isfile(fname)):
            n_iter = self.data.data['healpy']['n_iter_mcm']
            l_toeplitz, l_exact, dl_band = self.data.check_toeplitz('cls')
            if spin0:
                m1, m2 = self.get_mappers()
                msk1 = m1.get_mask()
                msk2 = m2.get_mask()
                f1 = nmt.NmtField(msk1, None, spin=0)
                f2 = nmt.NmtField(msk2, None, spin=0)
            else:
                f1, f2 = self.get_nmt_fields()
            w.compute_coupling_matrix(f1, f2, self.b, n_iter=n_iter,
                                      l_toeplitz=l_toeplitz, l_exact=l_exact,
                                      dl_band=dl_band)
            w.write_to(fname)
            self.recompute_mcmc = False
        else:
            w.read_from(fname)
        return w

    def get_cl_file(self):
        if self._read_symmetric:
            fname = os.path.join(self.outdir, f'cl_{self.tr2}_{self.tr1}.npz')
        else:
            fname = os.path.join(self.outdir, f'cl_{self.tr1}_{self.tr2}.npz')
        ell = self.b.get_effective_ells()
        recompute = self.recompute_cls or self.recompute_mcm
        if recompute or (not os.path.isfile(fname)):
            mapper1, mapper2 = self.get_mappers()
            f1, f2 = self.get_nmt_fields()
            w = self.get_workspace()
            wins = w.get_bandpower_windows()

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
                cl_cp = mapper1.get_cl_coupled()
                cl = w.decouple_cell(cl_cp)
                cls_cov = mapper1.get_cls_covar_coupled()
                cl_cov_cp = cls_cov['cross']
                cl_cov_11_cp = cls_cov['auto_11']
                cl_cov_12_cp = cls_cov['auto_12']
                cl_cov_22_cp = cls_cov['auto_22']
                # mapper.get_cl_coupled is assumed to return
                # the noise-less power spectrum. No need
                # to subtract it here.
            else:
                cl_cov_cp = nmt.compute_coupled_cell(f1, f2)
                cl_cov_11_cp = cl_cov_cp
                cl_cov_12_cp = cl_cov_cp
                cl_cov_22_cp = cl_cov_cp
                cl = w.decouple_cell(cl_cov_cp)
                cl_cp = cl_cov_cp - nl_cp
                cl -= nl
            np.savez(fname, ell=ell, cl=cl, cl_cp=cl_cp, nl=nl,
                     nl_cp=nl_cp, cl_cov_cp=cl_cov_cp,
                     cl_cov_11_cp=cl_cov_11_cp,
                     cl_cov_12_cp=cl_cov_12_cp,
                     cl_cov_22_cp=cl_cov_22_cp,
                     wins=wins)
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
                        'auto_11': ['cl_cov_11_cp'],
                        'auto_12': ['cl_cov_12_cp'],
                        'auto_22': ['cl_cov_22_cp']}

        return cl_file

    def get_ell_nl(self):
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.nl

    def get_ell_nl_cp(self):
        if self.nl_cp is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.nl_cp

    def get_ell_cl_cp(self):
        """
        Return the noisless coupled Cell
        """
        if self.ell is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.cl_cp

    def get_ell_cl_cp_cov(self):
        if self.ell is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.cls_cov['cross']

    def get_ell_cls_cp_cov_auto(self):
        if self.ell is None:
            self.get_cl_file()
        cl11 = self.cls_cov['auto_11']
        cl12 = self.cls_cov['auto_12']
        cl22 = self.cls_cov['auto_22']
        ell = np.arange(3 * self.nside)
        return ell, cl11, cl12, cl22

    def get_masks(self):
        mapper1, mapper2 = self.get_mappers()
        m1 = mapper1.get_mask()
        m2 = mapper2.get_mask()
        return m1, m2

    def get_masks_names(self):
        mapper1, mapper2 = self.get_mappers()
        m1 = mapper1.mask_name
        m2 = mapper2.mask_name
        return m1, m2

    def get_bandpower_windows(self):
        if self.ell is None:
            self.get_cl_file()
        return self.wins


class ClFid(ClBase):
    def __init__(self, data, tr1, tr2):
        super().__init__(data, tr1, tr2)
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
        self.cosmo = self.get_cosmo_ccl()
        self.hm_par = self._get_halomodel_params()
        self._ccl_tr1 = None
        self._ccl_tr2 = None

    def _get_halomodel_params(self):
        if 'halo_model' not in self.data.data['cov']['fiducial']:
            self.data.data['cov']['fiducial']['halo_model'] = {}
        hmp = self.data.data['cov']['fiducial']['halo_model']

        if 'mass_def' not in hmp:
            md = ccl.halos.MassDef200m()
        else:
            mds = hmp['mass_def']
            Delta = float(mds[:-1])
            if mds[-1] == 'm':
                rho_type = 'matter'
            elif mds[-1] == 'c':
                rho_type = 'critical'
            else:
                raise ValueError("Unknown density type %s" % (mds[-1]))
            md = ccl.halos.MassDef(Delta, rho_type)

        mfc = ccl.halos.mass_function_from_name(hmp.get('mass_function',
                                                        'Tinker10'))
        mf = mfc(self.cosmo, mass_def=md)

        hbc = ccl.halos.halo_bias_from_name(hmp.get('halo_bias',
                                                    'Tinker10'))
        hb = hbc(self.cosmo, mass_def=md)

        # We also need an NFW profile to handle certain cases
        cmc = ccl.halos.concentration_from_name(hmp.get('concentration',
                                                        'Duffy08'))
        cm = cmc(mdef=md)
        pNFW = ccl.halos.HaloProfileNFW(cm)

        # Halo model calculator
        hmc = ccl.halos.HMCalculator(self.cosmo, mf, hb, md)

        # Transition smoothing
        alpha = hmp.get('alpha_HMCODE', 0.7)
        # Small-k damping
        klow = hmp.get('k_suppress', 0.01)

        return {'mass_def': md,
                'mass_func': mf,
                'halo_bias': hb,
                'cM': cm,
                'prof_NFW': pNFW,
                'calculator': hmc,
                'alpha': (lambda a: alpha),
                'k_suppress': (lambda a: klow)}

    def get_cosmo_ccl(self):
        fiducial = self.data.data['cov']['fiducial']
        cosmo = ccl.Cosmology(**fiducial['cosmo'])
        return cosmo

    def get_tracers_ccl(self):
        if self._ccl_tr1 is None:
            mapper1, mapper2 = self.get_mappers()
            self._ccl_tr1 = self.compute_tracer_ccl(self.tr1, mapper1)
            self._ccl_tr2 = self.compute_tracer_ccl(self.tr2, mapper2)
        return self._ccl_tr1, self._ccl_tr2

    def compute_tracer_ccl(self, tr, mapper):
        dtype = mapper.get_dtype()
        tracer = self.data.data['tracers'][tr]
        fiducial = self.data.data['cov']['fiducial']
        ccl_pr = self.hm_par['prof_NFW']
        ccl_pr_2pt = None
        with_hm = tracer.get('use_halo_model', False)
        normed_profile = True
        # Get Tracers
        if dtype == 'galaxy_density':
            # Import z, pz
            z, pz = mapper.get_nz(dz=0)
            bias = (z, np.ones_like(z))
            # Get tracer
            ccl_tr = ccl.NumberCountsTracer(self.cosmo, has_rsd=False,
                                            dndz=(z, pz), bias=bias)
            if with_hm:
                hod_pars = tracer.get('hod_params', {})
                ccl_pr = ccl.halos.HaloProfileHOD(self.hm_par['cM'],
                                                  **hod_pars)
                ccl_pr_2pt = ccl.halos.Profile2ptHOD()
        elif dtype == 'galaxy_shear':
            # Import z, pz
            z, pz = mapper.get_nz(dz=0)
            # # Calculate bias IA
            ia_bias = None
            if fiducial['wl_ia']:
                # TODO: Improve this in yml file
                A, eta, z0 = fiducial['wl_ia']
                # pyccl2 -> has already the factor inside. Only needed bz
                bz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474
                ia_bias = (z, bz)
            # Get tracer
            ccl_tr = ccl.WeakLensingTracer(self.cosmo, dndz=(z, pz),
                                           ia_bias=ia_bias)
        elif dtype == 'cmb_convergence':
            # TODO: correct z_source
            ccl_tr = ccl.CMBLensingTracer(self.cosmo, z_source=1100)
        elif dtype == 'cmb_tSZ':
            normed_profile = False
            ccl_tr = ccl.tSZTracer(self.cosmo, z_max=3.)
            if with_hm:
                pars = tracer.get('gnfw_params', {})
                ccl_pr = ccl.halos.HaloProfilePressureGNFW(**pars)
        else:
            raise ValueError('Type of tracer not recognized. It can be \
                             galaxy_density, galaxy_shear, cmb_tSZ, or \
                             cmb_convergence!')
        return {'name': tr,
                'ccl_tr': ccl_tr,
                'ccl_pr': ccl_pr,
                'ccl_pr_2pt': ccl_pr_2pt,
                'with_hm': with_hm,
                'normed': normed_profile}

    def _get_ccl_pk(self, ccl_tr1, ccl_tr2):
        if ccl_tr1['with_hm'] or ccl_tr2['with_hm']:
            if ccl_tr1['name'] == ccl_tr2['name']:
                pr2pt = ccl_tr1['ccl_pr_2pt']
            else:
                pr2pt = None
            k_s = np.geomspace(1E-4, 1E2, 512)
            lk_s = np.log(k_s)
            a_s = 1./(1+np.linspace(0., 6., 30)[::-1])

            pk = ccl.halos.halomod_Pk2D(self.cosmo,
                                        self.hm_par['calculator'],
                                        ccl_tr1['ccl_pr'],
                                        prof_2pt=pr2pt,
                                        prof2=ccl_tr2['ccl_pr'],
                                        normprof1=ccl_tr1['normed'],
                                        normprof2=ccl_tr2['normed'],
                                        lk_arr=lk_s, a_arr=a_s)
            # We comment this out for now because these features are
            # not present in the pip release of CCL
            # smooth_transition=self.hm_par['alpha'],
            # supress_1h=self.hm_par['k_suppress'])
        else:
            pk = None
        return pk

    def _get_ccl_cl(self, ccl_tr1, ccl_tr2, ell):
        pk = self._get_ccl_pk(ccl_tr1, ccl_tr2)
        return ccl.angular_cl(self.cosmo,
                              ccl_tr1['ccl_tr'],
                              ccl_tr2['ccl_tr'],
                              ell, p_of_k_a=pk)

    def get_cl_file(self):
        nside = self.data.data['healpy']['nside']
        if self._read_symmetric:
            fname = os.path.join(self.outdir, f'cl_{self.tr2}_{self.tr1}.npz')
        else:
            fname = os.path.join(self.outdir, f'cl_{self.tr1}_{self.tr2}.npz')
        ell = np.arange(3 * nside)
        if not os.path.isfile(fname):
            ccl_tr1, ccl_tr2 = self.get_tracers_ccl()
            cl = self._get_ccl_cl(ccl_tr1, ccl_tr2, ell)
            b1 = self.data.data['tracers'][self.tr1].get('bias', 1.)
            b2 = self.data.data['tracers'][self.tr2].get('bias', 1.)
            cl *= b1*b2
            tracers = self.data.data['tracers']
            fiducial = self.data.data['cov']['fiducial']
            d1, d2 = self.get_dtypes()
            for dtype, tr in zip([self.tr1, self.tr2], [d1, d2]):
                if (dtype == 'galaxy_shear') and fiducial['wl_m']:
                    cl = (1 + tracers[tr]['m']) * cl

            # This is only valid for LCDM and spins 0 and 2.
            s1, s2 = self.get_spins()
            size = s1 + s2
            if size == 0:
                size = 1
            cl_vector = np.zeros((size, cl.size))
            cl_vector[0] = cl

            np.savez_compressed(fname, cl=cl_vector, ell=ell)

        cl_file = np.load(fname)
        if np.any(cl_file['ell'] != ell):
            raise ValueError(f'The ell in {fname} does not match the ell \
                               from nside={nside}')

        self.ell = cl_file['ell']
        self.cl = cl_file['cl']
        return cl_file


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
