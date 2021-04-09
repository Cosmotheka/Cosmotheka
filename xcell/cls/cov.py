#!/usr/bin/python
from .cl import Cl, ClFid
from .common import Data
import os
import numpy as np
import pymaster as nmt


class Cov():
    def __init__(self, data, trA1, trA2, trB1, trB2):
        self.data = Data(data=data)
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
        self.clfid_A1B1 = ClFid(data, trA1, trB1)
        self.clfid_A1B2 = ClFid(data, trA1, trB2)
        self.clfid_A2B1 = ClFid(data, trA2, trB1)
        self.clfid_A2B2 = ClFid(data, trA2, trB2)
        self.recompute_cov = self.data.data['recompute']['cov']
        self.recompute_cmcm = self.data.data['recompute']['cmcm']
        self.cov = None
        # Noise marginalization?
        self.nl_marg = False
        if (trA1 == trA2 == trB1 == trB2):
            trconf = self.data.data['tracers'][trA1]
            self.nl_marg = trconf.get('nl_marginalize', False)
            self.nl_prior = trconf.get('nl_prior', 1E30)

    def get_outdir(self):
        root = self.data.data['output']
        outdir = os.path.join(root, 'cov')
        return outdir

    def get_covariance_workspace(self):
        mask1, mask2 = self.clA1A2.get_masks_names()
        mask3, mask4 = self.clB1B2.get_masks_names()
        fname = os.path.join(self.outdir,
                             f'cw__{mask1}__{mask2}__{mask3}__{mask4}.fits')
        cw = nmt.NmtCovarianceWorkspace()
        recompute = self.data.data['recompute']['cmcm']
        if recompute or (not os.path.isfile(fname)):
            n_iter = self.data.data['healpy']['n_iter_cmcm']
            l_toeplitz, l_exact, dl_band = self.data.check_toeplitz('cov')
            fA1, fB1 = self.clA1B1.get_nmt_fields()
            fA2, fB2 = self.clA2B2.get_nmt_fields()
            cw.compute_coupling_coefficients(fA1, fA2, fB1, fB2,
                                             n_iter=n_iter,
                                             l_toeplitz=l_toeplitz,
                                             l_exact=l_exact,
                                             dl_band=dl_band)
            cw.write_to(fname)
            self.recompute_cmcm = False
        else:
            cw.read_from(fname)

        return cw

    def get_covariance(self, spin0=False):
        fname = os.path.join(self.outdir,
                             'cov_{}_{}_{}_{}.npz'.format(self.trA1,
                                                          self.trA2,
                                                          self.trB1,
                                                          self.trB2))
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

        # Weight the Cls
        cla1b1 = (cla1b1 + nla1b1) / np.mean(m_a1 * m_b1)
        cla1b2 = (cla1b2 + nla1b2) / np.mean(m_a1 * m_b2)
        cla2b1 = (cla2b1 + nla2b1) / np.mean(m_a2 * m_b1)
        cla2b2 = (cla2b2 + nla2b2) / np.mean(m_a2 * m_b2)
        #####
        wa = self.clA1A2.get_workspace(spin0=spin0)
        wb = self.clB1B2.get_workspace(spin0=spin0)
        cw = self.get_covariance_workspace()

        s_a1, s_a2 = self.clA1A2.get_spins()
        s_b1, s_b2 = self.clB1B2.get_spins()
        if spin0 and (s_a1 + s_a2 + s_b1 + s_b2 != 0):
            cov_e = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                            [cla1b1[0]], [cla1b2[0]],
                                            [cla2b1[0]], [cla2b2[0]],
                                            wa, wb)
            cov_b = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                            [cla1b1[-1]], [cla1b2[-1]],
                                            [cla2b1[-1]], [cla2b2[-1]],
                                            wa, wb)
            nbpw_a, nbpw_b = cov_e.shape
            # 00, 02
            if (s_a1 + s_a2 == 0) and (s_b1 + s_b2 == 2):
                nclsa = 1
                nclsb = 2
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[0, :, 1, :] = cov_b
            # 02, 00
            if (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 0):
                nclsa = 2
                nclsb = 1
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[1, :, 0, :] = cov_b
            # 00, 22
            if (s_a1 + s_a2 == 0) and (s_b1 + s_b2 == 4):
                nclsa = 1
                nclsb = 4
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[0, :, 3, :] = cov_b
            # 22, 00
            if (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 0):
                nclsa = 4
                nclsb = 1
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[3, :, 0, :] = cov_b
            # 02, 02
            if (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 2):
                nclsa = 2
                nclsb = 2
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[1, :, 1, :] = cov_b
            # 02, 22
            if (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 4):
                nclsa = 2
                nclsb = 4
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[1, :, 3, :] = cov_b
            # 22, 02
            if (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 2):
                nclsa = 4
                nclsb = 2
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[3, :, 1, :] = cov_b
            # 22, 22
            if (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 4):
                nclsa = 4
                nclsb = 4
                cov = np.zeros([nclsa, nbpw_a, nclsb, nbpw_b])
                cov[0, :, 0, :] = cov_e
                cov[1, :, 1, :] = cov_b
                cov[2, :, 2, :] = cov_b
                cov[3, :, 3, :] = cov_b
            cov = cov.reshape([nclsa*nbpw_a, nclsb*nbpw_b])
        else:
            cov = nmt.gaussian_covariance(cw, s_a1, s_a2, s_b1, s_b2,
                                          cla1b1, cla1b2, cla2b1, cla2b2,
                                          wa, wb)

        if self.nl_marg:
            _, nl = self.clA1A2.get_ell_nl()
            nl = nl.flatten()
            cov += self.nl_prior**2 * (nl[:, None] * nl[None, :])

        self.cov = cov
        np.savez_compressed(fname, cov=cov)
        self.recompute_cov = False
        return cov


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from \
                                     data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('trA1', type=str, help='Tracer A1 name')
    parser.add_argument('trA2', type=str, help='Tracer A2 name')
    parser.add_argument('trB1', type=str, help='Tracer B1 name')
    parser.add_argument('trB2', type=str, help='Tracer B2 name')
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    cov = Cov(data, args.trA1, args.trA2, args.trB1, args.trB2)
    cov.get_covariance()
