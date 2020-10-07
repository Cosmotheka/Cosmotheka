#!/usr/bin/python
from cl import Cl
from cov import Cov
import common as co
import numpy as np
import sacc
import os
import warnings

# TODO: move this to data.ylm?

class sfile():
    def __init__(self, datafile, output, use_nl=False):
        self.datafile = datafile
        self.data = co.read_data(datafile)
        self.outdir = self.data['output']
        self.use_nl = use_nl
        self.s = sacc.Sacc()
        self.add_tracers()
        self.add_ell_cls()
        self.add_covariance()
        fname = os.path.join(self.outdir, output)
        self.s.save_fits(fname, overwrite=True)

    def add_tracers(self):
        tracers = co.get_tracers_used(self.data)
        for tr in tracers:
            self.add_tracer(tr)

    def add_ell_cls(self):
        cl_tracers = co.get_cl_tracers(self.data)
        for tr1, tr2 in cl_tracers:
            self.add_ell_cl(tr1, tr2)

    def add_covariance(self):
        if self.use_nl:
            warnings.warn('Adding covariance matrix with use_nl=True is not yet implemented')
            return
        # Get nbpw
        dtype = self.s.get_data_types()[0]
        tracers = self.s.get_tracer_combinations(data_type=dtype)[0]
        ell, _ = self.s.get_ell_cl(dtype, *tracers)
        nbpw = ell.size
        #
        ndim = self.s.mean.size
        cl_tracers = self.s.get_tracer_combinations()

        covmat = -1 * np.ones((ndim, ndim))
        for i, trs1 in enumerate(cl_tracers):
            dof1 = co.get_dof_tracers(self.data, trs1)
            dtypes1 = self.get_datatypes_from_dof(dof1)
            for trs2 in cl_tracers[i:]:
                dof2 = co.get_dof_tracers(self.data, trs2)
                dtypes2 = self.get_datatypes_from_dof(dof2)
                print(trs1, trs2)
                cov = Cov(self.datafile, *trs1, *trs2).cov.reshape((nbpw, dof1, nbpw, dof2))

                for i, dt1 in enumerate(dtypes1):
                    ix1 = self.s.indices(tracers=trs1, data_type=dt1)
                    if len(ix1) == 0:
                        continue
                    for j, dt2 in enumerate(dtypes2):
                        ix2 = self.s.indices(tracers=trs2, data_type=dt2)
                        if len(ix2) == 0:
                            continue
                        covi = cov[:, i, :, j]
                        covmat[np.ix_(ix1, ix2)] = covi
                        covmat[np.ix_(ix2, ix1)] = covi.T

        self.s.add_covariance(covmat)

    def add_tracer(self, tr):
        tracer = self.data['tracers'][tr]

        if tracer['type'] == 'gc':
            quantity = 'galaxy_density'
            z, nz = np.loadtxt(tracer['dndz'], usecols=(1, 3), unpack=True)
            self.s.add_tracer('NZ', tr, quantity=quantity, spin=tracer['spin'],
                              z=z, nz=nz)
        elif tracer['type'] == 'wl':
            quantity = 'galaxy_shear'
            z, nz = np.loadtxt(tracer['dndz'], usecols=(1, 3), unpack=True)
            self.s.add_tracer('NZ', tr, quantity=quantity, spin=tracer['spin'],
                              z=z, nz=nz)
        elif tracer['type'] == 'cv':
            quantity = 'cmb_convergence'
            ell, nl = np.loadtxt(tracer['nl'], usecols=(0, 1), unpack=True)
            beam = np.ones_like(ell)
            self.s.add_tracer('Map', tr, quantity=quantity, spin=tracer['spin'],
                              ell=ell, beam=beam, beam_extra={'nl': nl})
        else:
            raise ValueError('Tracer type {} not implemented'.format(tracer['type']))


    def add_ell_cl(self, tr1, tr2):
        ells_nobin = np.arange(3 * self.data['healpy']['nside'])
        cl = Cl(self.datafile, tr1, tr2)
        w = cl.get_workspace()
        ws_bpw= w.get_bandpower_windows()

        wins = sacc.BandpowerWindow(ells_nobin, ws_bpw[0, :, 0, :].T)

        cl_types = self.get_datatypes_from_dof(cl.cl.shape[0])

        for i, cl_type in enumerate(cl_types):
            if self.use_nl:
                cli = cl.nl[i]
            else:
                cli = cl.cl[i]
            if (cl_type == 'cl_be') and (tr1 == tr2):
                continue

            self.s.add_ell_cl(cl_type, tr1, tr2, cl.ell, cli, window=wins)

    def get_datatypes_from_dof(self, dof):
        if dof  == 1:
            cl_types = ['cl_00']
        elif dof == 2:
            cl_types = ['cl_0e', 'cl_0b']
        elif dof == 4:
            cl_types = ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']
        else:
            raise ValueError('dof does not match 1, 2, or 4.')

        return cl_types

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('name', type=str, help="Name of the generated sacc file. Stored in yml['output']")
    parser.add_argument('--use_nl', action='store_true', default=False, help="Set if you want to use nl and covNG (if present) instead of cls and covG")
    args = parser.parse_args()

    sfile = sfile(args.INPUT, args.name, args.use_nl)
