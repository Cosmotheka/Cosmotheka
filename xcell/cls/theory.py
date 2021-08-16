import pyccl as ccl
import numpy as np


class Theory():
    def __init__(self, data):
        self.config = data['cov']['fiducial']
        self.cosmo = None
        self.get_cosmo_ccl()
        self.hm_par = self.get_halomodel_params()

    def get_cosmo_ccl(self):
        if self.cosmo is None:
            self.cosmo = ccl.Cosmology(**(self.config['cosmo']))
        return self.cosmo

    def get_halomodel_params(self):
        if self.cosmo is None:
            self.get_cosmo_ccl()

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

    def compute_tracer_ccl(self, name, tracer, mapper):
        dtype = mapper.get_dtype()
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
            if self.config['wl_ia']:
                # TODO: Improve this in yml file
                A, eta, z0 = self.config['wl_ia']
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
        return {'name': name,
                'ccl_tr': ccl_tr,
                'ccl_pr': ccl_pr,
                'ccl_pr_2pt': ccl_pr_2pt,
                'with_hm': with_hm,
                'normed': normed_profile}

    def get_ccl_tkka(self, ccl_trA1, ccl_trA2, ccl_trB1, ccl_trB2,
                     kind='1h'):
        if kind not in ['1h']:
            raise NotImplementedError(f"Non-Gaussian term {kind} "
                                      "not supported.")

        pA1 = ccl_trA1['ccl_pr']
        pA2 = ccl_trA2['ccl_pr']
        if ccl_trA1['name'] == ccl_trA2['name']:
            pr2ptA = ccl_trA1['ccl_pr_2pt']
            pA2 = pA1
        else:
            pr2ptA = None
        pB1 = ccl_trB1['ccl_pr']
        pB2 = ccl_trB2['ccl_pr']
        if ccl_trB1['name'] == ccl_trB2['name']:
            pr2ptB = ccl_trB1['ccl_pr_2pt']
            pB2 = pB1
        else:
            pr2ptB = None

        k_s = np.geomspace(1E-4, 1E2, 512)
        lk_s = np.log(k_s)
        a_s = 1./(1+np.linspace(0., 6., 30)[::-1])
        tkk = ccl.halos.halomod_Tk3D_1h(self.cosmo, self.hm_par['calculator'],
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

            pk = ccl.halos.halomod_Pk2D(self.cosmo,
                                        self.hm_par['calculator'],
                                        p1, prof_2pt=pr2pt, prof2=p2,
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

    def get_ccl_cl(self, ccl_tr1, ccl_tr2, ell):
        pk = self.get_ccl_pk(ccl_tr1, ccl_tr2)
        return ccl.angular_cl(self.cosmo,
                              ccl_tr1['ccl_tr'],
                              ccl_tr2['ccl_tr'],
                              ell, p_of_k_a=pk)

    def get_ccl_cl_covNG(self, ccl_trA1, ccl_trA2, ellA,
                         ccl_trB1, ccl_trB2, ellB, fsky,
                         kind='1h'):
        tkk = self.get_ccl_tkka(ccl_trA1, ccl_trA2,
                                ccl_trB1, ccl_trB2,
                                kind=kind)
        return ccl.angular_cl_cov_cNG(self.cosmo,
                                      cltracer1=ccl_trA1['ccl_tr'],
                                      cltracer2=ccl_trA2['ccl_tr'],
                                      ell=ellA,
                                      tkka=tkk, fsky=fsky,
                                      cltracer3=ccl_trB1['ccl_tr'],
                                      cltracer4=ccl_trB2['ccl_tr'],
                                      ell2=ellB)
