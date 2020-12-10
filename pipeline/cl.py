#!/usr/bin/python
from pyclbr import readmodule
from glob import glob
import common as co
import numpy as np
import healpy as hp
import pyccl as ccl
import pymaster as nmt
import os
import yaml
import warnings

class Cl():
    def __init__(self, data, tr1, tr2):
        self.data = data
        self.read_symm = False
        if ((tr1, tr2) not in co.get_cl_trs_names(self.data)) and \
           ((tr2, tr1) in co.get_cl_trs_names(self.data)):
            warnings.warn('Reading the symmetric element.')
            self.read_symm = True
            self.tr1 = tr2
            self.tr2 = tr1
        else:
            self.tr1 = tr1
            self.tr2 = tr2
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.nside = self.data['healpy']['nside']
        self.b = self.get_NmtBin()
        self.recompute_cls = self.data['recompute']['cls']
        self.recompute_mcm = self.data['recompute']['mcm']
        # Not needed to load cl if already computed
        self._mapper1 = None
        self._mapper2 = None
        self._w = None
        ##################
        self.ell = None
        self.cl = None
        self.nl = None
        self.nl_cp = None

    def get_outdir(self):
        root = self.data['output']
        trreq = ''.join(s for s in (self.tr1 + '_' + self.tr2) if not s.isdigit())
        outdir = os.path.join(root, trreq)
        return outdir

    def get_NmtBin(self):
        trs = self.tr1 + '-' + self.tr2
        trs = ''.join(s for s in trs if not s.isdigit())
        if 'bpw_edges' in self.data['cls'][trs].keys():
            bpw_edges = self.data['cls'][trs]['bpw_edges']
        else:
            bpw_edges = np.array(self.data['bpw_edges'])
        nside = self.nside
        bpw_edges = bpw_edges[bpw_edges <= 3 * nside] # 3*nside == ells[-1] + 1
        if 3*nside not in bpw_edges: # Exhaust lmax --> gives same result as previous method, but adds 1 bpw (not for 4096)
            bpw_edges = np.append(bpw_edges, 3*nside)
        b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
        return b

    def _get_mapper(self, tr):
        config = self.data['tracers'][tr]
        mapper_class = config['mapper_class']
        modules = glob('mappers/*.py')
        for m in modules:
            module_name = 'mappers.' + os.path.basename(m)
            module_name = module_name.replace('.py', '')
            try:
                clsmembers = readmodule(module_name)
            except ImportError:
                continue

            if mapper_class in clsmembers:
                exec(f'from {module_name} import {mapper_class}')
                mapper = exec(f'{mapper_class}({config})')
                return mapper

        raise ValueError('Mapper {} not implemented.'.format(mapper_class))

    def get_mappers(self):
        if self._mapper1 is None:
            self._mapper1 = self._get_mapper(self.tr1)
            self._mapper2 = self._mapper1 if self.tr1 == self.tr2 else self._get_mapper(self.tr2)
        return self._mapper1, self._mapper2

    def get_nmt_fields(self):
        mapper1, mapper2 = self.get_mappers()
        f1 = mapper1.get_nmt_field()
        f2 = mapper1.get_nmt_field()
        return f1, f2

    def get_workspace(self):
        if self._w is None:
            self._w = self._compute_workspace()
        return self._w

    def _compute_workspace(self):
        mask1, mask2 = self.get_masks_names()
        fname = os.path.join(self.outdir, 'w__{}__{}.fits'.format(mask1, mask2))
        w = nmt.NmtWorkspace()
        if self.recompute_mcm or (not os.path.isfile(fname)):
            n_iter = self.data['healpy']['n_iter_mcm']
            f1, f2 = self.get_nmt_fields()
            w.compute_coupling_matrix(f1, f2, self.b, n_iter=n_iter)
            w.write_to(fname)
            self.recompute_mcmc = False
        else:
            w.read_from(fname)
        return w

    def get_cl_file(self):
        fname = os.path.join(self.outdir, 'cl_{}_{}.npz'.format(self.tr1, self.tr2))
        ell = self.b.get_effective_ells()
        recompute = self.recompute_cls or self.recompute_mcm
        if recompute or (not os.path.isfile(fname)):
            mapper1, mapper2 = self.get_mappers()
            f1, f2 = self.get_nmt_fields()
            w = self.get_workspace()
            cl = w.decouple_cell(nmt.compute_coupled_cell(f1, f2))
            nl_cp = np.zeros((cl.shape[0], 3 * self.nside))
            if self.tr1 == self.tr2:
                nl_cp[0] = nl_cp[-1] = mapper1.get_nl_coupled()
            nl = w.decouple_cell(nl_cp)
            np.savez(fname, ell=ell, cl=cl-nl, nl=nl, nl_cp=nl_cp)
            self.recompute_cls = False

        cl_file = np.load(fname)
        cl = cl_file['cl']
        if np.any(ell != cl_file['ell']):
            raise ValueError('The file {} does not have the expected bpw. Aborting!'.format(fname))

        self.cl_file = cl_file
        self.ell = cl_file['ell']
        self.cl = cl_file['cl']
        self.nl = cl_file['nl']
        self.nl_cp = cl_file['nl_cp']
        return cl_file

    def get_ell_cl(self):
        return self.ell, self.cl

    def get_ell_nl(self):
        return self.ell, self.nl

    def get_ell_nl_cp(self):
        return self.ell, self.nl_cp

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

    def get_spins(self):
        mapper1, mapper2 = self.get_mappers()
        s1 = mapper1.get_spin()
        s2 = mapper2.get_spin()
        return s1, s2


class Cl_fid():
    def __init__(self, data, tr1, tr2):
        self.data = data
        self.read_symm = False
        if ((tr1, tr2) not in co.get_cl_trs_names(self.data)) and \
           ((tr2, tr1) in co.get_cl_trs_names(self.data)):
            warnings.warn('Reading the symmetric element.')
            self.read_symm = True
            self.tr1 = tr2
            self.tr2 = tr1
        else:
            self.tr1 = tr1
            self.tr2 = tr2
        self.tr1 = tr1
        self.tr2 = tr2
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.cosmo = self.get_cosmo_ccl()
        self._ccl_tr1 = None
        self._ccl_tr2 = None
        self.cl_file = self.get_cl_file()
        self.ell = self.cl_file['ell']
        self.cl = self.cl_file['cl']

    def get_outdir(self):
        root = self.data['output']
        trreq = ''.join(s for s in (tr1 + '_' + tr2) if not s.isdigit())
        outdir = os.path.join(root, 'fiducial', trreq)
        return outdir


    def get_cosmo_ccl(self):
        fiducial = self.data['cov']['fiducial']
        cosmo = ccl.Cosmology(**fiducial['cosmo'])
        return cosmo

    def get_tracers_ccl(self):
        if self._ccl_tr1 is None:
            self._ccl_tr1 = self.compute_tracer_ccl(self.tr1)
            self._ccl_tr2 = self.compute_tracer_ccl(self.tr2)
        return self._ccl_tr1, self._ccl_tr2

    def compute_tracer_ccl(self, tr):
        tracer = self.data['tracers'][tr]
        fiducial = self.data['cov']['fiducial']
        # Get Tracers
        if tracer['type'] == 'gc':
            # Import z, pz
            z, pz = np.loadtxt(tracer['dndz'], usecols=tracer['dndz_cols'], unpack=True)
            # Calculate z bias
            dz = 0
            z_dz = z - dz
            # Set to 0 points where z_dz < 0:
            sel = z_dz >= 0
            z_dz = z_dz[sel]
            pz = pz[sel]
            # Calculate bias
            bias = None
            if fiducial['gc_bias'] is True:
                bias = (z, tracer['bias'] * np.ones_like(z))
            # Get tracer
            return ccl.NumberCountsTracer(self.cosmo, has_rsd=False,
                                          dndz=(z_dz, pz), bias=bias)
        elif tracer['type'] == 'wl':
            # Import z, pz
            z, pz = np.loadtxt(tracer['dndz'], usecols=tracer['dndz_cols'], unpack=True)
            # Calculate z bias
            dz = 0
            z_dz = z - dz
            # Set to 0 points where z_dz < 0:
            sel = z_dz >= 0
            z_dz = z_dz[sel]
            pz = pz[sel]
            # # Calculate bias IA
            ia_bias = None
            if fiducial['wl_ia']:
                A, eta, z0 = fiducial['wl_ia']  # TODO: Improve this in yml file
                bz = A*((1.+z)/(1.+z0))**eta*0.0139/0.013872474  # pyccl2 -> has already the factor inside. Only needed bz
                ia_bias = (z, bz)
            # Get tracer
            return ccl.WeakLensingTracer(self.cosmo, dndz=(z_dz, pz),
                                         ia_bias=ia_bias)
        elif tracer['type'] == 'cv':
            return ccl.CMBLensingTracer(self.cosmo, z_source=1100) #TODO: correct z_source
        else:
            raise ValueError('Type of tracer not recognized. It can be gc, wl or cv!')

    def get_cl_file(self):
        nside = self.data['healpy']['nside']
        fname = os.path.join(self.outdir, 'cl_{}_{}.npz'.format(tr1, tr2))
        ell = np.arange(3 * nside)
        if not os.path.isfile(fname):
            ccl_tr1, ccl_tr2 = self.get_tracers_ccl()
            cl = ccl.angular_cl(self.cosmo, ccl_tr1, ccl_tr2, ell)
            tracers = self.data['tracers']
            fiducial = self.data['cov']['fiducial']
            for tr in [self.tr1, self.tr2]:
                if (tracers[tr]['type'] == 'wl') and fiducial['wl_m']:
                    cl = (1 + tracers[tr]['m']) * cl

            s1, s2 = self.get_spins()
            size = s1 + s2
            if size == 0:
                size = 1
            cl_vector = np.zeros((size, cl.size))
            cl_vector[0] = cl

            np.savez_compressed(fname, cl=cl_vector, ell=ell)

        cl_file = np.load(fname)
        if np.any(cl_file['ell'] != ell):
            raise ValueError('The ell in {} does not match the ell from nside={}'.format(fname, nside))
        return cl_file

    def get_ell_cl(self):
        return self.ell, self.cl

    def get_spins(self):
        tracers = self.data['tracers']
        s1 = tracers[self.tr1]['spin']
        s2 = tracers[self.tr2]['spin']
        return int(s1), int(s2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('tr1', type=str, help='Tracer 1 name')
    parser.add_argument('tr2', type=str, help='Tracer 2 name')
    parser.add_argument('--fiducial', default=False, action='store_true', help='Compute the fiducial model Cl')
    args = parser.parse_args()

    data = co.read_data(args.INPUT)
    if args.fiducial:
        cl = Cl_fid(data, args.tr1, args.tr2)
    else:
        cl = Cl(data, args.tr1, args.tr2)
    cl.get_cl_file()
