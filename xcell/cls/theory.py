import pyccl as ccl
import numpy as np


class ConcentrationDuffy08M500c(ccl.halos.Concentration):
    """ Concentration-mass relation by Duffy et al. 2008
    (arXiv:0804.2486) extended to Delta = 500-critical.
    Args:
        mdef (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'Duffy08M500c'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08M500c, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = ccl.halos.MassDef(500, 'critical')

    def _check_mdef(self, mdef):
        if (mdef.Delta != 500) or (mdef.rho_type != 'critical'):
            return True
        return False

    def _setup(self):
        self.A = 3.67
        self.B = -0.0903
        self.C = -0.51

    def _concentration(self, cosmo, M, a):
        # Same as Eq. 4 of the paper.
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)


class Theory():
    def __init__(self, data):
        self.config = data['cov']['fiducial']
        self._cosmo = None
        self._hm_par = None

    def get_cosmo_ccl(self):
        if self._cosmo is None:
            self._cosmo = ccl.Cosmology(**(self.config['cosmo']))
        return self._cosmo

    def get_halomodel_params(self):
        if self._hm_par is not None:
            return self._hm_par

        cosmo = self.get_cosmo_ccl()

        if 'halo_model' not in self.config:
            self.config['halo_model'] = {}
        hmp = self.config['halo_model']

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
        mf = mfc(cosmo, mass_def=md)

        hbc = ccl.halos.halo_bias_from_name(hmp.get('halo_bias',
                                                    'Tinker10'))
        hb = hbc(cosmo, mass_def=md)

        # We also need an NFW profile to handle certain cases
        cmc = ccl.halos.concentration_from_name(hmp.get('concentration',
                                                        'Duffy08'))
        cm = cmc(mdef=md)
        pNFW = ccl.halos.HaloProfileNFW(cm)
        p2pt = ccl.halos.Profile2pt()

        # Halo model calculator
        hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)

        # Transition smoothing
        alpha = hmp.get('alpha_HMCODE', 0.7)
        # Small-k damping
        klow = hmp.get('k_suppress', 0.01)

        self._hm_par = {'mass_def': md,
                        'mass_func': mf,
                        'halo_bias': hb,
                        'cM': cm,
                        'prof_NFW': pNFW,
                        'prof_2pt': p2pt,
                        'calculator': hmc,
                        'alpha': (lambda a: alpha),
                        'k_suppress': (lambda a: klow)}
        return self._hm_par

    def compute_tracer_ccl(self, name, tracer, mapper):
        cosmo = self.get_cosmo_ccl()
        hm_par = self.get_halomodel_params()

        dtype = mapper.get_dtype()
        ccl_pr = hm_par['prof_NFW']
        ccl_pr_2pt = hm_par['prof_2pt']
        with_hm = tracer.get('use_halo_model', False)
        normed_profile = True
        # Get Tracers
        if dtype == 'galaxy_density':
            # Import z, pz
            z, pz = mapper.get_nz(dz=0)
            bias = (z, np.ones_like(z))
            mag_bias = None
            mag_s = tracer.get('magnif_s', None)
            if mag_s:
                mag_bias = (z, np.ones_like(z) * mag_s)
            # Get tracer
            ccl_tr = ccl.NumberCountsTracer(cosmo, has_rsd=False,
                                            dndz=(z, pz), bias=bias,
                                            mag_bias=mag_bias)
            if with_hm:
                hod_pars = tracer.get('hod_params', {})
                ccl_pr = ccl.halos.HaloProfileHOD(hm_par['cM'],
                                                  **hod_pars)
                ccl_pr_2pt = ccl.halos.Profile2ptHOD()
        elif dtype == 'galaxy_shear':
            # Import z, pz
            z, pz = mapper.get_nz(dz=0)
            # # Calculate bias IA
            ia_bias = None
            if self.config['wl_ia']:
                # TODO: Improve this in yml file
                A, eta, z0 = self.config['wl_ia']
                # pyccl2 -> has already the factor inside. Only needed bz
                bz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474
                ia_bias = (z, bz)
            # Get tracer
            ccl_tr = ccl.WeakLensingTracer(cosmo, dndz=(z, pz),
                                           ia_bias=ia_bias)
        elif dtype == 'cmb_convergence':
            # TODO: correct z_source
            ccl_tr = ccl.CMBLensingTracer(cosmo, z_source=1100)
        elif dtype == 'cmb_tSZ':
            normed_profile = False
            ccl_tr = ccl.tSZTracer(cosmo, z_max=3.)
            if with_hm:
                pars = tracer.get('gnfw_params', {})
                ccl_pr = ccl.halos.HaloProfilePressureGNFW(**pars)
        else:
            raise ValueError('Type of tracer not recognized. It can be \
                             galaxy_density, galaxy_shear, cmb_tSZ, or \
                             cmb_convergence!')
        return {'name': name,
                'ccl_tr': ccl_tr,
                'ccl_pr': ccl_pr,
                'ccl_pr_2pt': ccl_pr_2pt,
                'with_hm': with_hm,
                'normed': normed_profile}

    def get_ccl_tkka(self, ccl_trA1, ccl_trA2, ccl_trB1, ccl_trB2,
                     kind='1h'):
        # Returns trispectrum for one of the non-Gaussian covariance terms.
        if kind not in ['1h']:
            raise NotImplementedError(f"Non-Gaussian term {kind} "
                                      "not supported.")

        cosmo = self.get_cosmo_ccl()
        hm_par = self.get_halomodel_params()

        pA1 = ccl_trA1['ccl_pr']
        pA2 = ccl_trA2['ccl_pr']
        if ccl_trA1['name'] == ccl_trA2['name']:
            pr2ptA = ccl_trA1['ccl_pr_2pt']
            pA2 = pA1
        else:
            pr2ptA = hm_par['prof_2pt']
        pB1 = ccl_trB1['ccl_pr']
        pB2 = ccl_trB2['ccl_pr']
        if ccl_trB1['name'] == ccl_trB2['name']:
            pr2ptB = ccl_trB1['ccl_pr_2pt']
            pB2 = pB1
        else:
            pr2ptB = hm_par['prof_2pt']
        k_s = np.geomspace(1E-4, 1E2, 512)
        lk_s = np.log(k_s)
        a_s = 1./(1+np.linspace(0., 6., 30)[::-1])
        tkk = ccl.halos.halomod_Tk3D_1h(cosmo, hm_par['calculator'],
                                        prof1=pA1, prof2=pA2,
                                        prof12_2pt=pr2ptA,
                                        prof3=pB1, prof4=pB2,
                                        prof34_2pt=pr2ptB,
                                        normprof1=ccl_trA1['normed'],
                                        normprof2=ccl_trA2['normed'],
                                        normprof3=ccl_trB1['normed'],
                                        normprof4=ccl_trB2['normed'],
                                        a_arr=a_s, lk_arr=lk_s)
        return tkk

    def get_ccl_pk(self, ccl_tr1, ccl_tr2):
        cosmo = self.get_cosmo_ccl()
        hm_par = self.get_halomodel_params()

        if ccl_tr1['with_hm'] or ccl_tr2['with_hm']:
            p1 = ccl_tr1['ccl_pr']
            p2 = ccl_tr2['ccl_pr']
            if ccl_tr1['name'] == ccl_tr2['name']:
                pr2pt = ccl_tr1['ccl_pr_2pt']
                p2 = p1
            else:
                pr2pt = None
            k_s = np.geomspace(1E-4, 1E2, 512)
            lk_s = np.log(k_s)
            a_s = 1./(1+np.linspace(0., 6., 30)[::-1])

            pk = ccl.halos.halomod_Pk2D(cosmo,
                                        hm_par['calculator'],
                                        p1, prof_2pt=pr2pt, prof2=p2,
                                        normprof1=ccl_tr1['normed'],
                                        normprof2=ccl_tr2['normed'],
                                        lk_arr=lk_s, a_arr=a_s)
            # We comment this out for now because these features are
            # not present in the pip release of CCL
            # smooth_transition=hm_par['alpha'],
            # supress_1h=hm_par['k_suppress'])
        else:
            pk = None
        return pk

    def get_ccl_cl(self, ccl_tr1, ccl_tr2, ell):
        cosmo = self.get_cosmo_ccl()

        pk = self.get_ccl_pk(ccl_tr1, ccl_tr2)
        return ccl.angular_cl(cosmo,
                              ccl_tr1['ccl_tr'],
                              ccl_tr2['ccl_tr'],
                              ell, p_of_k_a=pk)

    def get_ccl_cl_covNG(self, ccl_trA1, ccl_trA2, ellA,
                         ccl_trB1, ccl_trB2, ellB, fsky,
                         kind='1h'):
        cosmo = self.get_cosmo_ccl()

        tkk = self.get_ccl_tkka(ccl_trA1, ccl_trA2,
                                ccl_trB1, ccl_trB2,
                                kind=kind)
        return ccl.angular_cl_cov_cNG(cosmo,
                                      cltracer1=ccl_trA1['ccl_tr'],
                                      cltracer2=ccl_trA2['ccl_tr'],
                                      ell=ellA,
                                      tkka=tkk, fsky=fsky,
                                      cltracer3=ccl_trB1['ccl_tr'],
                                      cltracer4=ccl_trB2['ccl_tr'],
                                      ell2=ellB, integration_method='spline')
