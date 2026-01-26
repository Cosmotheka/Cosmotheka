#!/usr/bin/python
from .cl import Cl, ClFid
from .data import Data
from .theory import Theory
from . import tools
import os
import numpy as np
import pymaster as nmt
import time
import healpy as hp


class Cov:
    """
    Covariance class. This is in charge of computing the covariance block for
    four given tracers.
    """

    def __init__(
        self, data, trA1, trA2, trB1, trB2, ignore_existing_yml=False
    ):
        """
        Parameters
        ----------
        data: dict
            The configuration dictionary (e.g. a read configuration yaml file).
        trA1: str
            First tracer of the first Cell
        trA2: str
            Second tracer of the first Cell
        trB1: str
            First tracer of the second Cell
        trB2: str
            Second tracer of the second Cell
        ignore_existing_yml: bool
            If True, ignore existing yaml in the output directory and use the
            input configuration. Otherwise, use the existing yaml.
        """
        self.data = Data(data=data, ignore_existing_yml=ignore_existing_yml)
        self.tmat = self.data.get_tracer_matrix()
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.trA1 = trA1
        self.trA2 = trA2
        self.trB1 = trB1
        self.trB2 = trB2
        cl_dic, clfid_dic = self._load_Cls()
        self.clA1A2 = cl_dic[(trA1, trA2)]
        self.clB1B2 = cl_dic[(trB1, trB2)]
        self.clA1B1 = cl_dic[(trA1, trB1)]
        self.clA1B2 = cl_dic[(trA1, trB2)]
        self.clA2B1 = cl_dic[(trA2, trB1)]
        self.clA2B2 = cl_dic[(trA2, trB2)]
        self.clfid_A1A2 = clfid_dic[(trA1, trA2)]
        self.clfid_B1B2 = clfid_dic[(trB1, trB2)]
        self.clfid_A1B1 = clfid_dic[(trA1, trB1)]
        self.clfid_A1B2 = clfid_dic[(trA1, trB2)]
        self.clfid_A2B1 = clfid_dic[(trA2, trB1)]
        self.clfid_A2B2 = clfid_dic[(trA2, trB2)]
        self.recompute_cov = self.data.data["recompute"]["cov"]
        self.recompute_cmcm = self.data.data["recompute"]["cmcm"]
        self.cov = None
        self.fsky = self.data.data["cov"].get("fsky_NG", None)  # fsky for SSC and NG term
        # Noise marginalization?
        self.nl_marg = False
        if trA1 == trA2 == trB1 == trB2:
            trconf = self.data.data["tracers"][trA1]
            self.nl_marg = trconf.get("nl_marginalize", False)
            self.nl_prior = trconf.get("nl_prior", 1e30)
        # Spin-0 approximation
        self.spin0 = self.data.data["cov"].get("spin0", False)
        # Multiplicative bias marginalization
        self.m_marg = self.data.data["cov"].get("m_marg", False)
        self.cw = None

        # Non-Gaussian covariance terms
        self.NG_config = self.data.data["cov"].get("non_Gaussian", {})
        self.SSC_config = self.data.data["cov"].get("SSC", {})
        self.do_NG = self.NG_config.get("compute", False)
        self.do_SSC = self.SSC_config.get("compute", False)
        self.th = None
        self.ccl_trA1 = None
        self.ccl_trA2 = None
        self.ccl_trB1 = None
        self.ccl_trB2 = None

    def _load_Cls(self):
        """
        Return the Cells needed to compute the covariance block

        Return
        ------
        cl_dic: dict
            Dictionary of the data Cell needed. The keys are tuples of pair of
            tracers.
        clfid_dic: dict
            Dictionary of the fiducial Cell needed. The keys are tuples of pair
            of tracers.
        """
        data = self.data.data
        trs_comb = [
            (self.trA1, self.trA2),
            (self.trB1, self.trB2),
            (self.trA1, self.trB1),
            (self.trA1, self.trB2),
            (self.trA2, self.trB1),
            (self.trA2, self.trB2),
        ]

        # Load data Cls
        cl_dic = {}
        for trs in trs_comb:
            if trs not in cl_dic.keys():
                cl_dic[trs] = Cl(data, *trs, ignore_existing_yml=True)

        # Load fiducial Cls
        clfid_dic = {}
        for trs in trs_comb:
            if trs not in clfid_dic.keys():
                # Cl computed from data if needed
                if self.tmat[trs]["clcov_from_data"]:
                    cl = cl_dic[trs]
                else:
                    # Try to compute the fiducial Cl
                    try:
                        cl = ClFid(data, *trs, ignore_existing_yml=True)
                    except NotImplementedError as e:
                        if self.data.data["cov"].get("data_fallback", False):
                            # If that fails (e.g. unknown data type)
                            # this will be computed from the data.
                            cl = cl_dic[trs]
                            self.tmat[trs]["clcov_from_data"] = True
                        else:
                            raise e
                clfid_dic[trs] = cl

        return cl_dic, clfid_dic

    def get_outdir(self):
        """
        Return the output directory for the covariance blocks and workspaces

        Return
        ------
        outdir: str
            Path to the output directory for the covariance blocks and
            workspaces
        """
        root = self.data.data["output"]
        outdir = os.path.join(root, "cov")
        return outdir

    def get_covariance_workspace(self, save_cw=True):
        """
        Return the covariance workspace needed to compute the Gaussian
        covariance block. If 'recompute' is not set in the configuration file,
        it will read it from the output directory if found.

        Parameters
        ----------
        save_cw: bool
            If True, save the covariance workspace

        Return
        ------
        cw: pymaster.NmtCovarianceWorkspace
            Covariance workspace to compute the Gaussian covariance block

        """
        if self.cw is not None:
            return self.cw

        mask1, mask2, mask3, mask4 = self.data.get_cwsp_masks_for_tracers(
            self.trA1, self.trA2, self.trB1, self.trB2
        )

        fname = os.path.join(
            self.outdir, f"cw__{mask1}__{mask2}__{mask3}__{mask4}.fits"
        )
        cw = nmt.NmtCovarianceWorkspace()
        recompute = self.data.data["recompute"]["cmcm"]
        if (not recompute) and os.path.isfile(fname):
            tools.read_wsp(cw, fname)
            if cw.wsp is not None:
                return cw

        l_toeplitz, l_exact, dl_band = self.data.check_toeplitz("cov")
        fA1, fB1 = self.clA1B1.get_nmt_fields()
        fA2, fB2 = self.clA2B2.get_nmt_fields()
        cw.compute_coupling_coefficients(
            fA1,
            fA2,
            fB1,
            fB2,
            l_toeplitz=l_toeplitz,
            l_exact=l_exact,
            dl_band=dl_band,
        )
        if save_cw:
            tools.save_wsp(cw, fname)
        self.recompute_cmcm = False
        self.cw = cw

        return self.cw

    def _get_cl_for_cov(self, clab, clab_fid):
        """
        Return the angular power spectra to use in the computation of the
        covariance.

        Parameters
        ----------
        clab: cosmotheka.cls.Cl
            Instance of the Cl class for fields `a1 and `b`.
        clab_fid: cosmotheka.cls.ClFid or cosmotheka.cls.Cl
            Instance of the ClFid class for fields `a1 and `b`. If it is not an
            instance of ClFid, the data Cell will be used for the covariance.

        Return
        ------
        clab: numpy.array
            Array of angular power spectra between fields `a` and `b` to use in
            the computation of the covariance. Its shape will be (ncls,
            3*nside).

        """
        mean_mamb = clab.get_mean_mamb()
        if not mean_mamb:
            cl_cp = np.zeros((clab.get_n_cls(), 3 * clab.nside))
        else:
            if isinstance(clab_fid, ClFid):  # Compute from theory
                cl_cp = clab_fid.get_ell_cl_cp()[1] + clab.get_ell_nl_cp()[1]
            else:  # Compute from data
                # In this case we've requested to compute this
                # C_ell from the data, so `clab_fid` is actually
                # a `Cl` object (not a `ClFid` object).
                cl_cp = clab_fid.get_ell_cl_cp_cov()[1]
            cl_cp = cl_cp / mean_mamb

        return cl_cp

    def _get_covariance_spin0_approx(
        self,
        cw,
        s_a1,
        s_a2,
        s_b1,
        s_b2,
        cla1b1,
        cla1b2,
        cla2b1,
        cla2b2,
        wa,
        wb,
    ):
        """
        Return the block Gaussian covariance under the spin-0 approximation.

        Parameters
        ----------
        cw: pymaster.NmtCovarianceWorkspace
            Covariance workspace to compute the Gaussian covariance block
        s_a1: int
            Spin of the field `a1`
        s_a2: int
            Spin of the field `a2`
        s_b1: int
            Spin of the field `b1`
        s_b2: int
            Spin of the field `b2`
        cla1b1: numpy.array
            Array of angular power spectra between fields `a1` and `b1` to use
            in the computation of the covariance. Its shape has to be (ncls,
            3*nside).
        cla1b2: numpy.array
            Array of angular power spectra between fields `a1` and `b2` to use
            in the computation of the covariance. Its shape has to be (ncls,
            3*nside).
        cla2b1: numpy.array
            Array of angular power spectra between fields `a2` and `b1` to use
            in the computation of the covariance. Its shape has to be (ncls,
            3*nside).
        cla2b2: numpy.array
            Array of angular power spectra between fields `a2` and `b2` to use
            in the computation of the covariance. Its shape has to be (ncls,
            3*nside).
        wa: pymaster.NmtWorkspace
            Workspace for the Cell between the fields a1 and a2
        wb: pymaster.NmtWorkspace
            Workspace for the Cell between the fields b1 and b2

        Return
        ------
        cov: numpy.array
            Block covariance
        """
        nbpw_a = wa.wsp.bin.n_bands
        nbpw_b = wb.wsp.bin.n_bands
        nclsa = np.max([1, s_a1 + s_a2])
        nclsb = np.max([1, s_b1 + s_b2])
        cov = np.zeros([nbpw_a, nclsa, nbpw_b, nclsb])
        # 00, 02
        if (s_a1 + s_a2 == 0) and (s_b1 + s_b2 == 2):
            c_tt_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_tt_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[-1]],
                [cla2b1[0]],
                [cla2b2[-1]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_tt_te
            cov[:, 0, :, 1] = c_tt_tb
        # 02, 00
        elif (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 0):
            c_te_tt = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_tb_tt = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[-1]],
                [cla2b2[-1]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_te_tt
            cov[:, 1, :, 0] = c_tb_tt
        # 00, 22
        elif (s_a1 + s_a2 == 0) and (s_b1 + s_b2 == 4):
            c_tt_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_tt_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[-1]],
                [cla2b1[0]],
                [cla2b2[-1]],
                wa,
                wb,
            )
            c_tt_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[0]],
                [cla2b1[-1]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_tt_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[-1]],
                [cla2b1[-1]],
                [cla2b2[-1]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_tt_ee
            cov[:, 0, :, 1] = c_tt_eb
            cov[:, 0, :, 2] = c_tt_be
            cov[:, 0, :, 3] = c_tt_bb
        # 22, 00
        elif (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 0):
            c_ee_tt = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_eb_tt = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[-1]],
                [cla2b2[-1]],
                wa,
                wb,
            )
            c_be_tt = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[-1]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_bb_tt = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[-1]],
                [cla2b1[-1]],
                [cla2b2[-1]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_ee_tt
            cov[:, 1, :, 0] = c_eb_tt
            cov[:, 2, :, 0] = c_be_tt
            cov[:, 3, :, 0] = c_bb_tt
        # 02, 02
        elif (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 2):
            c_te_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_te_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[-1]],
                [cla2b1[0]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_tb_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[-1]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_tb_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[-1]],
                [cla2b1[-1]],
                [cla2b2[3]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_te_te
            cov[:, 0, :, 1] = c_te_tb
            cov[:, 1, :, 0] = c_tb_te
            cov[:, 1, :, 1] = c_tb_tb
        # 02, 22
        elif (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 4):
            c_te_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_te_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[-1]],
                [cla2b1[0]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_te_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[0]],
                [cla2b1[1]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_te_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[-1]],
                [cla2b1[1]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_tb_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[2]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_tb_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[-1]],
                [cla2b1[2]],
                [cla2b2[3]],
                wa,
                wb,
            )
            c_tb_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[0]],
                [cla2b1[3]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_tb_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[-1]],
                [cla2b1[3]],
                [cla2b2[3]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_te_ee
            cov[:, 0, :, 1] = c_te_eb
            cov[:, 0, :, 2] = c_te_be
            cov[:, 0, :, 3] = c_te_bb
            cov[:, 1, :, 0] = c_tb_ee
            cov[:, 1, :, 1] = c_tb_eb
            cov[:, 1, :, 2] = c_tb_be
            cov[:, 1, :, 3] = c_tb_bb
        # 22, 02
        elif (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 2):
            c_ee_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_ee_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[1]],
                [cla2b1[0]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_eb_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[-1]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_eb_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[1]],
                [cla2b1[-1]],
                [cla2b2[3]],
                wa,
                wb,
            )
            c_be_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[2]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_be_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[3]],
                [cla2b1[0]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_bb_te = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[2]],
                [cla2b1[-1]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_bb_tb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[-1]],
                [cla1b2[3]],
                [cla2b1[-1]],
                [cla2b2[3]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_ee_te
            cov[:, 1, :, 0] = c_eb_te
            cov[:, 2, :, 0] = c_be_te
            cov[:, 3, :, 0] = c_bb_te
            cov[:, 0, :, 1] = c_ee_tb
            cov[:, 1, :, 1] = c_eb_tb
            cov[:, 2, :, 1] = c_be_tb
            cov[:, 3, :, 1] = c_bb_tb
        # 22, 22
        elif (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 4):
            c_ee_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_ee_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[1]],
                [cla2b1[0]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_ee_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[1]],
                [cla1b2[0]],
                [cla2b1[1]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_ee_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[1]],
                [cla1b2[1]],
                [cla2b1[1]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_eb_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[0]],
                [cla2b1[2]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_eb_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[0]],
                [cla1b2[1]],
                [cla2b1[2]],
                [cla2b2[3]],
                wa,
                wb,
            )
            c_eb_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[1]],
                [cla1b2[0]],
                [cla2b1[3]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_eb_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[1]],
                [cla1b2[1]],
                [cla2b1[3]],
                [cla2b2[3]],
                wa,
                wb,
            )
            c_be_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[2]],
                [cla1b2[2]],
                [cla2b1[0]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_be_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[2]],
                [cla1b2[3]],
                [cla2b1[0]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_be_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[3]],
                [cla1b2[2]],
                [cla2b1[1]],
                [cla2b2[0]],
                wa,
                wb,
            )
            c_be_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[3]],
                [cla1b2[3]],
                [cla2b1[1]],
                [cla2b2[1]],
                wa,
                wb,
            )
            c_bb_ee = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[2]],
                [cla1b2[2]],
                [cla2b1[2]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_bb_eb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[2]],
                [cla1b2[3]],
                [cla2b1[2]],
                [cla2b2[3]],
                wa,
                wb,
            )
            c_bb_be = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[3]],
                [cla1b2[2]],
                [cla2b1[3]],
                [cla2b2[2]],
                wa,
                wb,
            )
            c_bb_bb = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                [cla1b1[3]],
                [cla1b2[3]],
                [cla2b1[3]],
                [cla2b2[3]],
                wa,
                wb,
            )
            cov[:, 0, :, 0] = c_ee_ee
            cov[:, 0, :, 1] = c_ee_eb
            cov[:, 0, :, 2] = c_ee_be
            cov[:, 0, :, 3] = c_ee_bb
            cov[:, 1, :, 0] = c_eb_ee
            cov[:, 1, :, 1] = c_eb_eb
            cov[:, 1, :, 2] = c_eb_be
            cov[:, 1, :, 3] = c_eb_bb
            cov[:, 2, :, 0] = c_be_ee
            cov[:, 2, :, 1] = c_be_eb
            cov[:, 2, :, 2] = c_be_be
            cov[:, 2, :, 3] = c_be_bb
            cov[:, 3, :, 0] = c_bb_ee
            cov[:, 3, :, 1] = c_bb_eb
            cov[:, 3, :, 2] = c_bb_be
            cov[:, 3, :, 3] = c_bb_bb

        return cov.reshape([nclsa * nbpw_a, nclsb * nbpw_b])

    def get_covariance(self, save_cw=True):
        """
        Return the block covariance with all requested contributions.

        Parameters
        ----------
        save_cw: bool
            If True, save the covariance workspace

        Return
        ------
        cov: numpy.array
            Block covariance
        """
        # Prepare the covariance dictionary
        s_a1, s_a2 = self.clA1A2.get_spins()
        s_b1, s_b2 = self.clB1B2.get_spins()

        size1 = self.clA1A2.get_ell_cl()[1].size
        size2 = self.clB1B2.get_ell_cl()[1].size

        zeros = np.zeros((size1, size2))

        cov_dict = {'cov': zeros.copy(),
                    'cov_G': zeros.copy(),
                    'cov_NG': zeros.copy(),
                    'cov_SSC': zeros.copy(),
                    'cov_nl_marg': zeros.copy(),
                    'cov_m_marg': zeros.copy(),
                    'cov_NG_1h': zeros.copy(),
                    'cov_NG_2h': zeros.copy(),
                    'cov_NG_3h': zeros.copy(),
                    'cov_NG_4h': zeros.copy(),
                    'threshold': None,
                    'notnull': None
                    }

        # Check if the covariance has already been computed and load it.
        # It will check if any new term needs to be computed; e.g.
        # the first time you only requested cov_G, but now you also want
        # cov_SSC. To avoid recomputing cov_G, we will load it from file,
        # compute SSC and save it again.
        fname = os.path.join(
            self.outdir,
            "cov_{}_{}_{}_{}.npz".format(
                self.trA1, self.trA2, self.trB1, self.trB2
            ),
        )
        # TODO: recompute_cmcm should trigger only cov_G recomputation
        recompute = self.recompute_cov or self.recompute_cmcm
        write_npz = True  # We will save the covariance at the end
        if (not recompute) and os.path.isfile(fname):
            print(f"Loading covariance from {fname}", flush=True)
            covf = np.load(fname)
            # Check if there is overlap. If not, we can return directly,
            # as it will be 0.
            if not covf.get('notnull', True):
                self.cov = cov_dict['cov']
                return self.cov
            cov_dict.update(dict(covf))
            write_npz = False  # We will not overwrite the file unless needed

        notnull = cov_dict['notnull']
        # If notnull is None, it means that it hasn't been computed yet.
        # If it is not None, we check if it's True and if cov_G is empty.
        # As of now, cov_G is always computes, so it shouldn't be empty.
        if notnull is None or (notnull is True and 
                               not np.any(cov_dict['cov_G'])):
            print("Computing required C_ells for covariance and checking if "
                  "they are non-zero...", flush=True)
            # Compute weighted Cls
            # Check if it's the auto-covariance of an auto-correlation
            auto_auto = self.trA1 == self.trA2 == self.trB1 == self.trB2
            # Check if it must be computed from data
            aa_data = (
                auto_auto and self.tmat[(self.trA1, self.trA2)]["clcov_from_data"]
            )
            # If so, get these C_ells
            itime = time.time()
            if aa_data:
                mean_mamb = self.clA1B1.get_mean_mamb()
                _, cla1b1, cla1b2, cla2b2 = self.clA1B1.get_ell_cls_cp_cov_auto()
                cla2b2 = cla2b2 / mean_mamb
                cla2b1 = cla1b2 / mean_mamb
                cla1b2 = cla1b2 / mean_mamb
                cla1b1 = cla1b1 / mean_mamb
            else:
                cla1b1 = self._get_cl_for_cov(self.clA1B1, self.clfid_A1B1)
                cla1b2 = self._get_cl_for_cov(self.clA1B2, self.clfid_A1B2)
                cla2b1 = self._get_cl_for_cov(self.clA2B1, self.clfid_A2B1)
                cla2b2 = self._get_cl_for_cov(self.clA2B2, self.clfid_A2B2)
            ftime = time.time()
            print(f"Get C_ells. It took {(ftime - itime) / 60} min", flush=True)

            notnull = (
                np.any(cla1b1)
                or np.any(cla1b2)
                or np.any(cla2b1)
                or np.any(cla2b2)
            )
            cov_dict['notnull'] = notnull
            write_npz = True  # We will need to save the file again

        if notnull and not np.any(cov_dict['cov_G']):
            print('Computing Gaussian covariance...', flush=True)
            itime = time.time()
            wa = self.clA1A2.get_workspace_cov()
            wb = self.clB1B2.get_workspace_cov()
            ftime = time.time()
            print(
                f"Get workspaces. It took {(ftime - itime) / 60} min",
                flush=True,
            )

            itime = time.time()
            cw = self.get_covariance_workspace(save_cw=save_cw)
            ftime = time.time()
            print(
                "Get covariance workspace. It took "
                + f"{(ftime - itime) / 60} min",
                flush=True,
            )

            itime = time.time()
            if self.spin0 and (s_a1 + s_a2 + s_b1 + s_b2 != 0):
                cov_dict['cov_G'] = self._get_covariance_spin0_approx(
                    cw,
                    s_a1,
                    s_a2,
                    s_b1,
                    s_b2,
                    cla1b1,
                    cla1b2,
                    cla2b1,
                    cla2b2,
                    wa,
                    wb,
                )

            else:
                cov_dict['cov_G'] = nmt.gaussian_covariance(
                    cw,
                    s_a1,
                    s_a2,
                    s_b1,
                    s_b2,
                    cla1b1,
                    cla1b2,
                    cla2b1,
                    cla2b2,
                    wa,
                    wb,
                )
            ftime = time.time()
            print(
                "Computed Gaussian covariance. It took "
                + f"{(ftime - itime) / 60} min",
                flush=True,
            )
            write_npz = True  # We will need to save the file again

        if self.nl_marg and notnull and not np.any(cov_dict['cov_nl_marg']):
            print('Computing nl_marg covariance...', flush=True)
            itime = time.time()
            cov_dict['cov_nl_marg'] = self.get_covariance_nl_marg()
            ftime = time.time()
            print(
                f"Computed nl_marg. It took {(ftime - itime) / 60} min",
                flush=True,
            )
            write_npz = True  # We will need to save the file again

        if self.m_marg and notnull and not np.any(cov_dict['cov_m_marg']):
            print('Computing m_marg covariance...', flush=True)
            itime = time.time()
            cov_dict['cov_m_marg'] = self.get_covariance_m_marg()
            ftime = time.time()
            print(
                f"Computed m_marg. It took {(ftime - itime) / 60} min",
                flush=True,
            )
            write_npz = True  # We will need to save the file again

        if self.do_SSC and notnull and not np.any(cov_dict['cov_SSC']):
            print('Computing SSC...', flush=True)
            itime = time.time()
            cov_dict['cov_SSC'] = self.get_SSC_halomodel(s_a1, s_a2, s_b1, s_b2)
            ftime = time.time()
            print(
                f"Computed SSC. It took {(ftime - itime) / 60} min",
                flush=True,
            )
            write_npz = True  # We will need to save the file again

        if self.do_NG and notnull and not np.any(cov_dict['cov_NG']):
            print('Computing cNG covariance...', flush=True)
            kinds = self.NG_config.get("NG_terms", None)

            itime = time.time()
            if kinds is None:
                print("Computing the full cNG covariance", flush=True)
                cov_dict['cov_NG'] = self.get_covariance_ng_halomodel(s_a1, s_a2, s_b1, s_b2)
            else:
                for kind in kinds:
                    print(f"Computing the cNG covariance term: {kind}", flush=True)
                    itime_kind = time.time()
                    cov_dict[f'cov_NG_{kind}'] = self.get_covariance_ng_halomodel(
                        s_a1, s_a2, s_b1, s_b2, kind
                    )
                    cov_dict['cov_NG'] += cov_dict[f'cov_NG_{kind}']
                    ftime_kind = time.time()
                    print(
                        f"Computed NG covariance for {kind}. It took {(ftime_kind - itime_kind) / 60} min",
                        flush=True,
                    )
            ftime = time.time()
            print(
                f"Computed NG covariance. It took {(ftime - itime) / 60} min",
                flush=True,
            )
            write_npz = True  # We will need to save the file again

        itime = time.time()
        self.cov = cov_dict['cov_G'] + cov_dict['cov_nl_marg'] + cov_dict['cov_m_marg'] + cov_dict['cov_SSC'] + cov_dict['cov_NG']
        cov_dict['cov'] = self.cov
        #
        ftime = time.time()
        print(
            "Added all covariances terms. It took "
            + f"{(ftime - itime) / 60} min",
            flush=True,
        )

        if cov_dict['threshold'] is None:
            print("Setting error threshold...", flush=True)
            threshold = self.data.data["cov"].get("error_threshold", None)
            if threshold is None:
                print("Computing error threshold...", flush=True)
                err1 = np.max(self.clA1A2.get_ell_cl_crude_error()[1])
                err2 = np.max(self.clB1B2.get_ell_cl_crude_error()[1])
                # Use order of magnitude
                # Squaring the errors to compare cov vs sigma^2, not sigma
                threshold = 10 ** int(np.log10(np.max([err1, err2]) ** 2)) * 1e5
            cov_dict['threshold'] = threshold
            write_npz = True  # We will need to save the file again

        if write_npz:
            itime = time.time()
            tools.save_npz(
                fname,
                **cov_dict,
            )
            ftime = time.time()
            print(
                f"Saved cov npz file. It took {(ftime - itime) / 60} min",
                flush=True,
            )
        else:
            print("Covariance npz file up to date.", flush=True)
        self.recompute_cov = False
        return self.cov

    def get_fsky(self):
        if self.fsky is not None:
            return self.fsky

        # Load all masks once
        itime = time.time()
        m_a1, m_a2 = self.clA1A2.get_masks()
        m_b1, m_b2 = self.clB1B2.get_masks()
        ftime = time.time()
        print(
            f"Masks read. It took {(ftime - itime) / 60} min",
            flush=True,
        )

        fsky = np.mean(
            ((m_a1 > 0) & (m_a2 > 0) & (m_b1 > 0) & (m_b2 > 0))
        )
        self.fsky = fsky

        return fsky

    def _get_thoery(self):
        if self.th is not None:
            return self.th

        self.th = Theory(self.data.data)
        return self.th

    def _get_CCL_tracers(self):
        if self.ccl_trA1 is not None:
            return (
                self.ccl_trA1,
                self.ccl_trA2,
                self.ccl_trB1,
                self.ccl_trB2,
            )

        mpA1, mpA2 = self.clA1A2.get_mappers()
        mpB1, mpB2 = self.clB1B2.get_mappers()
        trlist = self.data.data["tracers"]
        th = self._get_thoery()
        self.ccl_trA1 = th.compute_tracer_ccl(self.trA1, trlist[self.trA1], mpA1)
        self.ccl_trA2 = th.compute_tracer_ccl(self.trA2, trlist[self.trA2], mpA2)
        self.ccl_trB1 = th.compute_tracer_ccl(self.trB1, trlist[self.trB1], mpB1)
        self.ccl_trB2 = th.compute_tracer_ccl(self.trB2, trlist[self.trB2], mpB2)

        return self.ccl_trA1, self.ccl_trA2, self.ccl_trB1, self.ccl_trB2

    def get_covariance_ng_halomodel(
        self, s_a1, s_a2, s_b1, s_b2, kind=None
    ):
        """
        Return the block non-Guassian covariance under the halo model.

        Parameters
        ----------
        s_a1: int
            Spin of the field `a1`
        s_a2: int
            Spin of the field `a2`
        s_b1: int
            Spin of the field `b1`
        s_b2: int
            Spin of the field `b2`
        kind: str or None
            Kind of NG covariance to compute. If None, compute the full NG
            covariance. Else, halo model term: '1h', '2h', '3h' or '4h' for
            the 1-, 2-, 3- and 4-halo terms.

        Return
        ------
        cov: numpy.array
            Block covariance
        """
        ellA = self.clA1A2.b.get_effective_ells()
        ellB = self.clB1B2.b.get_effective_ells()
        nclsa = np.max([1, s_a1 + s_a2])
        nclsb = np.max([1, s_b1 + s_b2])
        cov = np.zeros([ellA.size, nclsa, ellB.size, nclsb])
        th = self._get_thoery()
        ccl_trA1, ccl_trA2, ccl_trB1, ccl_trB2 = self._get_CCL_tracers()
        fsky = self.get_fsky()
        covNG = th.get_ccl_cl_covNG(
            ccl_trA1, ccl_trA2, ellA, ccl_trB1, ccl_trB2, ellB, fsky, kind
        )
        # NG covariances can only be calculated for E-modes
        # bA1 = self.data.get_bias(self.trA1)
        # bA2 = self.data.get_bias(self.trA2)
        # bB1 = self.data.get_bias(self.trB1)
        # bB2 = self.data.get_bias(self.trB2)
        # cov[:, 0, :, 0] = covNG * bA1 * bA2 * bB1 * bB2

        cov[:, 0, :, 0] = covNG  # bias already included via HOD
        return cov.reshape([ellA.size * nclsa, ellB.size * nclsb])

    def get_SSC_halomodel(
        self, s_a1, s_a2, s_b1, s_b2
    ):
        """
        Return the block non-Guassian covariance under the halo model.

        Parameters
        ----------
        s_a1: int
            Spin of the field `a1`
        s_a2: int
            Spin of the field `a2`
        s_b1: int
            Spin of the field `b1`
        s_b2: int
            Spin of the field `b2`

        Return
        ------
        cov: numpy.array
            Block covariance
        """
        ellA = self.clA1A2.b.get_effective_ells()
        ellB = self.clB1B2.b.get_effective_ells()
        nclsa = np.max([1, s_a1 + s_a2])
        nclsb = np.max([1, s_b1 + s_b2])
        cov = np.zeros([ellA.size, nclsa, ellB.size, nclsb])
        th = self._get_thoery()
        ccl_trA1, ccl_trA2, ccl_trB1, ccl_trB2 = self._get_CCL_tracers()
        # We need the biases for the "linear_bias" approx
        bA1 = self.data.get_bias(self.trA1)
        bA2 = self.data.get_bias(self.trA2)
        bB1 = self.data.get_bias(self.trB1)
        bB2 = self.data.get_bias(self.trB2)
        
        sigma2_B_method = self.SSC_config.get("sigma2_B", "mask_wl")

        mask_wl = None
        fsky = None
        if sigma2_B_method == "mask_wl":
            print("Using mask window function method for SSC", flush=True)
            m_a1, m_a2 = self.clA1A2.get_masks()
            m_b1, m_b2 = self.clB1B2.get_masks()
            m12 = m_a1 * m_a2
            m34 = m_b1 * m_b2
            area = hp.nside2pixarea(hp.npix2nside(m12.size))

            alm = hp.map2alm(m12)
            blm = hp.map2alm(m34)

            mask_wl = hp.alm2cl(alm, blm)
            mask_wl *= 2 * np.arange(mask_wl.size) + 1
            mask_wl /= np.sum(m12) * np.sum(m34) * area**2
        elif sigma2_B_method == "fsky":
            print("Using fsky method for SSC", flush=True)
            fsky = self.get_fsky()
        else:
            raise ValueError(
                "Unknown sigma2_B_method for SSC: {}".format(
                    sigma2_B_method
                )
            )

        covSSC = th.get_ccl_cl_covSSC(
            ccl_trA1, ccl_trA2, ellA, ccl_trB1, ccl_trB2, ellB,
            bA1, bA2, bB1, bB2, mask_wl=mask_wl, fsky=fsky
        )
        # NG covariances can only be calculated for E-modes
        cov[:, 0, :, 0] = covSSC
        return cov.reshape([ellA.size * nclsa, ellB.size * nclsb])

    def get_covariance_nl_marg(self):
        """
        Return the analytically marginalized noise block covariance.

        Return
        ------
        cov: numpy.array
            Block covariance
        """
        _, nl = self.clA1A2.get_ell_nl()
        nl = nl.flatten()
        if self.trA1 == self.trA2 == self.trB1 == self.trB2:
            cov = self.nl_prior**2 * (nl[:, None] * nl[None, :])
        else:
            cov = np.zeros((nl.size, nl.size))

        return self._order_cov_as_NaMaster(cov)

    def get_covariance_m_marg(self):
        """
        Return the analytically marginalized multiplicative bias block
        covariance.

        Return
        ------
        cov: numpy.array
            Block covariance
        """
        # Check first if the covariance is not going to be 0's
        # to avoid reading and multiplying by the window function
        t_a1, t_a2 = self.clA1A2.get_dtypes()
        t_b1, t_b2 = self.clB1B2.get_dtypes()
        #
        sigma_a1 = sigma_a2 = sigma_b1 = sigma_b2 = 0
        if t_a1 == "galaxy_shear":
            sigma_a1 = self.data.data["tracers"][self.trA1].get("sigma_m", 0)
        if t_a2 == "galaxy_shear":
            sigma_a2 = self.data.data["tracers"][self.trA2].get("sigma_m", 0)
        if t_b1 == "galaxy_shear":
            sigma_b1 = self.data.data["tracers"][self.trB1].get("sigma_m", 0)
        if t_b2 == "galaxy_shear":
            sigma_b2 = self.data.data["tracers"][self.trB2].get("sigma_m", 0)

        factor = (
            sigma_a1 * sigma_b1
            + sigma_a1 * sigma_b2
            + sigma_a2 * sigma_b1
            + sigma_a2 * sigma_b2
        )

        if factor == 0:
            _, cla1a2 = self.clA1A2.get_ell_cl()
            _, clb1b2 = self.clB1B2.get_ell_cl()
            return np.zeros((cla1a2.size, clb1b2.size))

        _, cla1a2 = self.clfid_A1A2.get_ell_cl()
        _, clb1b2 = self.clfid_B1B2.get_ell_cl()
        # Window convolution only needed if computed from theory
        if isinstance(self.clfid_A1A2, ClFid):
            wins_a1a2 = self.clA1A2.get_bandpower_windows()
            ncls_a1a2, nell_cp = cla1a2.shape
            wins_a1a2 = wins_a1a2.reshape((-1, ncls_a1a2 * nell_cp))
            cla1a2 = wins_a1a2.dot(cla1a2.flatten()).reshape((ncls_a1a2, -1))
        if isinstance(self.clfid_B1B2, ClFid):
            wins_b1b2 = self.clB1B2.get_bandpower_windows()
            ncls_b1b2, nell_cp = clb1b2.shape
            wins_b1b2 = wins_b1b2.reshape((-1, ncls_b1b2 * nell_cp))
            clb1b2 = wins_b1b2.dot(clb1b2.flatten()).reshape((ncls_b1b2, -1))
        #
        cov = cla1a2.flatten()[:, None] * clb1b2.flatten()[None, :]
        cov *= factor
        return self._order_cov_as_NaMaster(cov)

    def _order_cov_as_NaMaster(self, cov):
        """
        Return a covariance block ordered as NaMaster. It assumes the input
        covariance is ordered as `Cell[:, None] * Cell[None, :]`.

        Parameters
        ----------
        cov: numpy.array
            Block covariance ordered as `Cell[:, None] * Cell[None, :]`

        Return
        ------
        cov: numpy.array
            Block covariance ordered as NaMaster
        """
        _, cla1a2 = self.clA1A2.get_ell_cl()
        _, clb1b2 = self.clB1B2.get_ell_cl()
        ncls_a1a2, nbpw = cla1a2.shape
        ncls_b1b2, _ = clb1b2.shape

        shape = cov.shape
        cov = cov.reshape((ncls_a1a2, nbpw, ncls_b1b2, nbpw))
        cov = np.moveaxis(cov, (0, 1, 2, 3), (1, 0, 3, 2))

        return cov.reshape(shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Cls and cov from \
                                     data.yml file"
    )
    parser.add_argument("INPUT", type=str, help="Input YAML data file")
    parser.add_argument("trA1", type=str, help="Tracer A1 name")
    parser.add_argument("trA2", type=str, help="Tracer A2 name")
    parser.add_argument("trB1", type=str, help="Tracer B1 name")
    parser.add_argument("trB2", type=str, help="Tracer B2 name")
    parser.add_argument(
        "--m_marg",
        default=False,
        action="store_true",
        help="Compute multiplicative bias marginalization cov",
    )
    parser.add_argument(
        "--nl_marg",
        default=False,
        action="store_true",
        help="Compute noise bias marginalization cov",
    )
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    cov = Cov(data, args.trA1, args.trA2, args.trB1, args.trB2)

    if args.m_marg:
        cov.get_covariance_m_marg()
    elif args.nl_marg:
        cov.get_covariance_nl_marg()
    else:
        cov.get_covariance()
