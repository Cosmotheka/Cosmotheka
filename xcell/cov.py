#!/usr/bin/python
from cl import Cl, Cl_fid
import os
import yaml
import common as co
import numpy as np
import healpy as hp
import pymaster as nmt

class Cov():
    def __init__(self, data, trA1, trA2, trB1, trB2):
        self.data = data
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.trA1 = trA1
        self.trA2 = trA2
        self.trB1 = trB1
        self.trB2 = trB2
        self.clA1A2 = Cl(data, trA1, trA2)
        self.clB1B2 = Cl(data, trB1, trB2)
        self.clA1B1 = Cl(data, trA1, trB1)
        self.clA1B2 = Cl(data, trA1, trB2)
        self.clA2B1 = Cl(data, trA2, trB1)
        self.clA2B2 = Cl(data, trA2, trB2)
        self.clfid_A1B1 = Cl_fid(data, trA1, trB1)
        self.clfid_A1B2 = Cl_fid(data, trA1, trB2)
        self.clfid_A2B1 = Cl_fid(data, trA2, trB1)
        self.clfid_A2B2 = Cl_fid(data, trA2, trB2)
        self.recompute_cov = self.data['recompute']['cov']
        self.recompute_cmcm = self.data['recompute']['cmcm']
        self.cov = None

    def get_outdir(self):
        root = self.data['output']
        outdir = os.path.join(root, 'cov')
        return outdir

    def get_covariance_workspace(self):
        mask1, mask2 = self.clA1A2.get_masks_names()
        mask3, mask4 = self.clB1B2.get_masks_names()
        fname = os.path.join(self.outdir, 'cw__{}__{}__{}__{}.fits'.format(mask1, mask2, mask3, mask4))
        cw = nmt.NmtCovarianceWorkspace()
        recompute = self.data['recompute']['cmcm']
        if recompute or (not os.path.isfile(fname)):
            n_iter = self.data['healpy']['n_iter_cmcm']
            l_toeplitz, l_exact, dl_band = co.check_toeplitz(self.data, 'cov')
            fA1, fB1 = self.clA1B1.get_nmt_fields()
            fA2, fB2 = self.clA2B2.get_nmt_fields()
            cw.compute_coupling_coefficients(fA1, fA2, fB1, fB2,
                                             n_iter=n_iter,
                                      l_toeplitz=l_toeplitz, l_exact=l_exact, dl_band=dl_band)
            cw.write_to(fname)
            self.recompute_cmcm = False
        else:
            cw.read_from(fname)

        return cw

    def get_covariance(self):
        fname = os.path.join(self.outdir, 'cov_{}_{}_{}_{}.npz'.format(self.trA1, self.trA2,
                                                                       self.trB1, self.trB2))
        recompute = self.recompute_cov or self.recompute_cmcm
        if (not recompute) and os.path.isfile(fname):
            self.cov = np.load(fname)['cov']
            return self.cov

        wa1b1 = self.clA1B1.get_workspace()
        wa1b2 = self.clA1B2.get_workspace()
        wa2b1 = self.clA2B1.get_workspace()
        wa2b2 = self.clA2B2.get_workspace()

        # Couple Theory Cls
        cla1b1 = wa1b1.couple_cell(self.clfid_A1B1.get_ell_cl()[1])
        cla1b2 = wa1b2.couple_cell(self.clfid_A1B2.get_ell_cl()[1])
        cla2b1 = wa2b1.couple_cell(self.clfid_A2B1.get_ell_cl()[1])
        cla2b2 = wa2b2.couple_cell(self.clfid_A2B2.get_ell_cl()[1])
        #####
        nla1b1 = self.clA1B1.get_ell_nl_cp()[1]
        nla1b2 = self.clA1B2.get_ell_nl_cp()[1]
        nla2b1 = self.clA2B1.get_ell_nl_cp()[1]
        nla2b2 = self.clA2B2.get_ell_nl_cp()[1]
        #####
        m_a1, m_b1 = self.clA1B1.get_masks()
        m_a2, m_b2 = self.clA2B2.get_masks()

        ##### Weight the Cls
        cla1b1 = (cla1b1 + nla1b1) / np.mean(m_a1 * m_b1)
        cla1b2 = (cla1b2 + nla1b2) / np.mean(m_a1 * m_b2)
        cla2b1 = (cla2b1 + nla2b1) / np.mean(m_a2 * m_b1)
        cla2b2 = (cla2b2 + nla2b2) / np.mean(m_a2 * m_b2)
        #####
        wa = self.clA1A2.get_workspace()
        wb = self.clB1B2.get_workspace()
        cw = self.get_covariance_workspace()

        s_a1, s_a2 = self.clA1A2.get_spins()
        s_b1, s_b2 = self.clB1B2.get_spins()

        cov = nmt.gaussian_covariance(cw, s_a1, s_a2, s_b1, s_b2,
                                      cla1b1, cla1b2, cla2b1, cla2b2,
                                      wa, wb)

        self.cov = cov
        np.savez_compressed(fname, cov=cov)
        self.recompute_cov = False
        return cov

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('trA1', type=str, help='Tracer A1 name')
    parser.add_argument('trA2', type=str, help='Tracer A2 name')
    parser.add_argument('trB1', type=str, help='Tracer B1 name')
    parser.add_argument('trB2', type=str, help='Tracer B2 name')
    args = parser.parse_args()

    data = co.read_data(args.INPUT)
    cov = Cov(data, args.trA1, args.trA2, args.trB1, args.trB2)
    cov.get_covariance()
