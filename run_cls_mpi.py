#!/usr/bin/python
import os
from cosmotheka.cls.data import Data
from cosmotheka.cls.cl import Cl, ClFid
from cosmotheka.cls.cov import Cov
from cosmotheka.cls.to_sacc import ClSack
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def check_skip(data, skip, trs):
    if skip is None:
        return False

    for tr in trs:
        if tr in skip:
            return True
        elif data.get_tracer_bare_name(tr) in skip:
            return True
    return False


def launch_cls(data, fiducial=False, skip=None):
    """
    Launch the computation of Cls for all tracers in data.
    If fiducial is True, compute the fiducial Cls.
    """
    cl_tracers = data.get_cl_trs_names()
    cl_tracers_per_wsp = data.get_cl_tracers_per_wsp()

    if RANK == 0:
        print(
            f"Computing Cls for {len(cl_tracers)} tracer pairs...", flush=True
        )

    my_cl_jobs = [
        cl_tracers_per_wsp[keys]
        for i, keys in enumerate(cl_tracers_per_wsp.keys())
        if i % SIZE == RANK
    ]

    for cl_tracers_with_wsp in my_cl_jobs:
        wsp = None
        for tr1, tr2 in cl_tracers_with_wsp:
            if check_skip(data, skip, [tr1, tr2]):
                print(
                    f"[Rank {RANK}] Skipping Cl for {tr1}, {tr2} as requested.",
                    flush=True,
                )
                continue
            fname = os.path.join(
                data.data["output"],
                data.get_tracers_bare_name_pair(tr1, tr2, "_"),
                f"cl_{tr1}_{tr2}.npz",
            )
            recompute = (
                data.data["recompute"]["cls"] or data.data["recompute"]["mcm"]
            )
            if os.path.isfile(fname) and not recompute:
                print(
                    f"[Rank {RANK}] Cl for {tr1}, {tr2} already exists, skipping.",
                    flush=True,
                )
                continue
            print(f"[Rank {RANK}] Computing Cl for {tr1}, {tr2}", flush=True)

            if fiducial:
                cl = ClFid(data.data, tr1, tr2)
            else:
                cl = Cl(data.data, tr1, tr2)
                # Avoid reading the workspace if it is already computed
                cl._w = wsp
            cl.get_cl_file()
            if wsp is None and isinstance(cl, Cl):
                wsp = cl.get_workspace()

    print(f"[Rank {RANK}] Cl computation finished.", flush=True)
    COMM.Barrier()


def launch_cov(data, skip=[]):
    """
    Launch the computation of Covariance blocks for all tracers in data.
    """
    cov_tracers = data.get_cov_trs_names()
    cov_tracers_per_cwsp = data.get_cov_tracers_per_cwsp()

    if RANK == 0:
        print(
            f"Computing Covariance blocks for {len(cov_tracers)} tracer pairs...",
            flush=True,
        )

    my_cov_jobs = [
        cov_tracers_per_cwsp[keys]
        for i, keys in enumerate(cov_tracers_per_cwsp.keys())
        if i % SIZE == RANK
    ]

    for cov_tracers_with_wsp in my_cov_jobs:
        cwsp = None
        for trs in cov_tracers_with_wsp:
            if check_skip(data, skip, trs):
                print(
                    f"[Rank {RANK}] Skipping Cov for {trs} as requested.",
                    flush=True,
                )
                continue
            fname = os.path.join(
                data.data["output"],
                "cov",
                "cov_{}_{}_{}_{}.npz".format(*trs),
            )
            recompute = (
                data.data["recompute"]["cov"] or data.data["recompute"]["cmcm"]
            )
            if os.path.isfile(fname) and not recompute:
                print(
                    f"[Rank {RANK}] Cov for {trs} already exists, skipping.",
                    flush=True,
                )
                continue
            print(f"[Rank {RANK}] Computing Cov for {trs}", flush=True)

            cov = Cov(data.data, *trs)
            # Avoid reading the workspace if it is already computed
            cov.cw = cwsp
            cov.get_covariance()

            if cwsp is None:
                cwsp = cov.get_covariance_workspace()

    print(f"[Rank {RANK}] Covariance computation finished.")
    COMM.Barrier()


def launch_to_sacc(data, fname, use, m_marg):
    """
    Launch the conversion of Cls and Covariance blocks to Sacc format.
    If use is 'nl', use the noise covariance instead of the Cls.
    If use is 'fiducial', use the fiducial Cls instead of the data Cls.
    """
    if RANK == 0:
        print(f"Converting to Sacc format using {use}...", flush=True)

        sacc = ClSack(data, fname, use, m_marg)

    COMM.Barrier()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Cls and cov from data.yml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("INPUT", type=str, help="Input YAML data file")
    parser.add_argument(
        "compute",
        type=str,
        choices=["cls", "cov", "to_sacc"],
        help="Compute: cls, cov or to_sacc.",
    )
    parser.add_argument(
        "--to_sacc_name",
        type=str,
        default="cls_cov.fits",
        help="Sacc file name",
    )
    parser.add_argument(
        "--to_sacc_use_nl",
        default=False,
        action="store_true",
        help="Set if you want to use nl and cov extra (if present) instead of cls and covG ",
    )
    parser.add_argument(
        "--to_sacc_use_fiducial",
        default=False,
        action="store_true",
        help="Set if you want to use the fiducial Cl and covG instead of data cls",
    )
    parser.add_argument(
        "--cls_fiducial",
        default=False,
        action="store_true",
        help="Set to compute the fiducial cls",
    )
    parser.add_argument(
        "--skip",
        default=[],
        nargs="+",
        help="Skip the following tracers. It can be given as DELS__0 to skip \
            only DELS__0 tracer or DELS to skip all DELS tracers",
    )
    parser.add_argument(
        "--override_yaml",
        default=False,
        action="store_true",
        help="Override the YAML file if already stored. Be ware that this \
            could cause compatibility problems in your data!",
    )
    parser.add_argument(
        "--to_sacc_m_marg",
        default=False,
        action="store_true",
        help="Set if you want to use store the covariance for the maginalized \
              multiplicative bias.",
    )

    args = parser.parse_args()

    ##############################################################################

    data = Data(data_path=args.INPUT, override=args.override_yaml)

    # 1. Compute Cells
    launch_cls(data, fiducial=args.cls_fiducial, skip=args.skip)

    if args.compute == "cls":
        if RANK == 0:
            print("Cls computation finished.")
        exit(0)

    # 2. Compute Covariance
    if not args.to_sacc_use_nl:
        launch_cov(data, skip=args.skip)

        if args.compute == "cov":
            if RANK == 0:
                print("Covariance computation finished.")
            exit(0)

    # 3. Convert to Sacc
    if args.to_sacc_use_nl and args.to_sacc_use_fiducial:
        raise ValueError(
            "Only one of --to_sacc_use_nl or --to_sacc_use_fiducial can be set"
        )
    elif args.to_sacc_use_nl:
        use = "nl"
    elif args.to_sacc_use_fiducial:
        use = "fiducial"
    else:
        use = "cls"

    m_marg = args.to_sacc_m_marg == "m_marg"
    launch_to_sacc(
        data.data_path, fname=args.to_sacc_name, use=use, m_marg=m_marg
    )

    if RANK == 0:
        print("Sacc compilation finished.")
    exit(0)
