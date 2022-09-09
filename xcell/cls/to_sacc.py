#!/usr/bin/python
from .cl import Cl, ClFid
from .cov import Cov
from .data import Data
import numpy as np
import sacc
import os


class ClSack():
    """
    ClSack class. Use it to generate a sacc file with the requested Cell and
    covariance matrix.
    """
    def __init__(self, datafile, output, use='cls', m_marg=False):
        """
        Parameters
        ----------
        datafile: str
            The path to the configuration yaml file
        output: str
            The sacc file name
        use: str
            One of 'cls', 'nl' or 'fiducial'. It generates, respectively, a
            sacc file with the data cls, the noise or the fiducial cls.
        m_marg: bool
            If True, the store the analytically marginalized multiplicative
            bias covariance.

        Raises
        ------
        ValueError
            If use is not one of 'cls', 'nl' or 'fiducial'
        """
        self.data = Data(data_path=datafile)
        self.outdir = self.data.data['output']
        self.use_nl = False
        self.use_fiducial = False
        if use == 'cls':
            pass
        elif use == 'nl':
            self.use_nl = True
        elif use == 'fiducial':
            self.use_fiducial = True
        else:
            raise ValueError('Use must be one of cls, nl or fiducial')
        self.m_marg = m_marg
        self.s = sacc.Sacc()
        self.add_tracers()
        self.add_ell_cls()
        self.add_covariance()
        fname = os.path.join(self.outdir, output)
        self.s.save_fits(fname, overwrite=True)

    def add_tracers(self):
        """
        Add the requested tracers to the sacc file
        """
        tracers = self.data.get_tracers_used()
        for tr in tracers:
            self.add_tracer(tr)

    def add_ell_cls(self):
        """
        Add the requested Cells to the sacc file
        """
        cl_tracers = self.data.get_cl_trs_names()
        for tr1, tr2 in cl_tracers:
            self.add_ell_cl(tr1, tr2)

    def read_covariance_extra(self):
        """
        Read the external covariance provided

        Raises
        ------
        ValueError:
            If the number of cls expecified in the configuration file for the
            external covariance does not match the number of computed cls.

        Return
        ------
        covariance: numpy.array
            External covariance
        """
        dtype = self.s.get_data_types()[0]
        cl_tracers = self.s.get_tracer_combinations(data_type=dtype)
        ell, _ = self.s.get_ell_cl(dtype, *cl_tracers[0])
        nbpw = ell.size
        #
        cl_extra_tracers = self.data.get_cov_extra_cl_tracers()
        # Read the extra covmat from file
        cov_extra = self.data.data['cov']['extra']
        cov = np.load(cov_extra['path'])
        ncls = int(cov.shape[0] / nbpw)
        cov = cov.reshape((ncls, nbpw, ncls, nbpw))
        if ('has_b' in cov_extra) and (cov_extra['has_b'] is True):
            raise NotImplementedError('Reading extra Cov with B-modes not ' +
                                      'implemented')

        else:
            if ncls != len(cl_extra_tracers):
                raise ValueError('Number of cls do not match')

        # Initialize the covmat that will go into the sacc file
        ndim = self.s.mean.size
        covmat = np.zeros((int(ndim/nbpw), nbpw, int(ndim/nbpw), nbpw))
        print(ndim/nbpw)
        cl_tracers = self.s.get_tracer_combinations()
        for i, trs1 in enumerate(cl_tracers):
            if trs1 not in cl_extra_tracers:
                continue
            ix1 = cl_extra_tracers.index(trs1)
            cl_ix1 = int(self.s.indices(tracers=trs1)[0] / nbpw)
            for j, trs2 in enumerate(cl_tracers[i:], i):
                if trs2 not in cl_extra_tracers:
                    continue
                ix2 = cl_extra_tracers.index(trs2)
                cl_ix2 = int(self.s.indices(tracers=trs2)[0] / nbpw)
                covi = cov[ix1, :, ix2, :]
                covmat[cl_ix1, :, cl_ix2, :] = covi
                covmat[cl_ix2, :, cl_ix1, :] = covi.T
        print(self.s.indices(tracers=trs1))
        return covmat.reshape((ndim, ndim))

    def add_covariance_extra(self):
        """
        Adds the external covariance to the sacc file
        """
        covmat = self.read_covariance_extra()
        self.s.add_covariance(covmat)

    def add_covariance_G(self, m_marg):
        """
        Adds the Gaussian covariance to the sacc file

        Parameters
        ----------
        m_marg: bool
            If True, add the analytically marginalized multiplicative bias
            covariance. Else, add the Gaussian covariance
        """
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
            dof1 = self.get_dof_tracers(trs1)
            dtypes1 = self.get_datatypes_from_dof(dof1)
            for trs2 in cl_tracers[i:]:
                dof2 = self.get_dof_tracers(trs2)
                dtypes2 = self.get_datatypes_from_dof(dof2)
                print(trs1, trs2)

                cov_class = Cov(self.data.data, *trs1, *trs2,
                                ignore_existing_yml=True)
                if m_marg:
                    cov = cov_class.get_covariance_m_marg()
                else:
                    cov = cov_class.get_covariance()
                cov = cov.reshape((nbpw, dof1, nbpw, dof2))

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

        # covmat += self.read_covariance_extra()
        self.s.add_covariance(covmat)

    def add_covariance(self):
        """
        Add the requested covariance to the sacc file
        """
        if self.use_nl:
            if self.m_marg:
                self.add_covariance_G(self.m_marg)
            elif 'extra' in self.data.data['cov']:
                self.add_covariance_extra()
        else:
            self.add_covariance_G(self.m_marg)

    def add_tracer(self, tr):
        """
        Add the input tracer to the sacc file

        Parameters
        ----------
        tr: str
            Tracer name of the tracer to be added.

        Raises
        ------
        NotImplementedError
            For tracers' quantities not implemented. The implemented ones are
            'galaxy_density', 'galaxy_shear', 'cmb_convergence', 'cmb_tSZ' and
            'generic'
        """
        mapper = self.data.get_mapper(tr)
        quantity = mapper.get_dtype()
        spin = mapper.get_spin()
        if quantity == 'galaxy_density':
            z, nz = mapper.get_nz(dz=0)
            self.s.add_tracer('NZ', tr, quantity=quantity, spin=spin,
                              z=z, nz=nz)
        elif quantity == 'galaxy_shear':
            z, nz = mapper.get_nz(dz=0)
            self.s.add_tracer('NZ', tr, quantity=quantity, spin=spin,
                              z=z, nz=nz)
        elif quantity in ['cmb_convergence', 'cmb_tSZ', 'generic']:
            ell = mapper.get_ell()
            nl = mapper.get_nl_coupled()[0]
            beam = mapper.get_beam()
            self.s.add_tracer('Map', tr, quantity=quantity, spin=spin,
                              ell=ell, beam=beam, beam_extra={'nl': nl})
        else:
            raise NotImplementedError(f'Tracer type {quantity} not ' +
                                      'implemented')

    def add_ell_cl(self, tr1, tr2):
        """
        Add the Cell corresponding to tracers tr1 and tr2

        Parameters
        ----------
        tr1: str
            First tracer's name
        tr2: str
            Second tracer's name
        """
        ells_nobin = np.arange(3 * self.data.data['sphere']['nside'])
        cl = Cl(self.data.data, tr1, tr2, ignore_existing_yml=True)
        ells_eff = cl.b.get_effective_ells()

        if self.use_fiducial:
            # bpw = np.array(self.data.data['bpw_edges'])
            # ells_eff = bpw[:-1] + np.diff(bpw)/2.
            cl = ClFid(self.data.data, tr1, tr2, ignore_existing_yml=True)
            ws_bpw = np.zeros((ells_eff.size, ells_nobin.size))
            ws_bpw[np.arange(ells_eff.size), ells_eff.astype(int)] = 1

        cl.get_cl_file()

        cl_types = self.get_datatypes_from_dof(cl.cl.shape[0])

        for i, cl_type in enumerate(cl_types):
            if (cl_type == 'cl_be') and (tr1 == tr2):
                continue
            elif self.use_nl:
                cli = cl.nl[i]
                wins = None
            elif self.use_fiducial:
                cli = ws_bpw.dot(cl.cl[i])
                wins = sacc.BandpowerWindow(ells_nobin, ws_bpw.T)
            else:
                wins = sacc.BandpowerWindow(ells_nobin, cl.wins[i, :, i, :].T)
                cli = cl.cl[i]

            self.s.add_ell_cl(cl_type, tr1, tr2, ells_eff, cli, window=wins)

    def get_dof_tracers(self, tracers):
        """
        Return the number of degrees of freedom for a pair of tracers

        Parameters
        ----------
        tracers: list
            List of pair of tracers

        Return
        ------
        dof: int
            Number of degrees of freedom
        """
        cl = Cl(self.data.data, tracers[0], tracers[1],
                ignore_existing_yml=True)
        return cl.get_n_cls()

    def get_datatypes_from_dof(self, dof):
        """
        Return a list of possible data types for a given number of degrees of
        freedom

        Parameters
        ----------
        dof: int
            Degrees of freedom

        Raises
        ------
        ValueError
            If dof is not one of 1, 2 or 4

        Return
        ------
        data_types: list
            List of data types for the given number of degrees of freedom
        """
        if dof == 1:
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
    parser = argparse.ArgumentParser(description="Compute Cls and cov from \
                                     data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('name', type=str, help="Name of the generated sacc \
                        file. Stored in yml['output']")
    parser.add_argument('--use_nl', action='store_true', default=False,
                        help="Set if you want to use nl and extra cov \
                        (if present) instead of cls and covG")
    parser.add_argument('--use_fiducial', action='store_true', default=False,
                        help="Set if you want to use the fiducial Cl and \
                        covG instead of data cls")
    parser.add_argument('--m_marg', action='store_true', default=False,
                        help="Set if you want to use store the covariance for \
                        the maginalized multiplicative bias.")
    args = parser.parse_args()

    if args.use_nl and args.use_fiducial:
        raise ValueError('Only one of --use_nl or --use_fiducial can be set')
    elif args.use_nl:
        use = 'nl'
    elif args.use_fiducial:
        use = 'fiducial'
    else:
        use = 'cls'

    sfile = ClSack(args.INPUT, args.name, use, args.m_marg)
