#!/usr/bin/python
from xcell.cls.data import Data
from xcell.cls.cov import Cov
import os


def run_cwsp_batch(data, trs_list):
    """
    Parameters
    ----------
    data (dict)
    trs_list (list): list of tracers for cov
    """
    ntrs =  len(trs_list)
    print("Runing covariance for tracers {} {} {} {} [1/{}]".format(*trs_list[0], ntrs))
    cov = Cov(data, *trs_list[0])
    cov.get_covariance()
    cw = cov.get_covariance_workspace()
    print()

    for i, trs in enumerate(trs_list[1:], 1):
        print("Runing covariance for tracers {} {} {} {} [{}/{}]".format(*trs, i+1, ntrs))
        cov = Cov(data, *trs)
        cov.cw = cw
        cov.get_covariance()
        print()


def check_same_cwsp(data, trs_list):
    cwsp = []
    for trs  in trs_list:
        mask1, mask2, mask3, mask4 = [data['tracers'][trsi]["mask_name"] for trsi in trs]
        fname = os.path.join(data['output'],
                             f'cov/cw__{mask1}__{mask2}__{mask3}__{mask4}.fits')
        if fname in cwsp:
            continue
        cwsp.append(fname)

    if len(cwsp) != 1:
        return False
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute the covariance "
                                     "blocks corresponding to the same "
                                     "coariance workspace")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('-trs', type=str, nargs='+', action='append',
                        help='Tracers for a covariance block (i.e. 4 tracers)')
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    trs_list = args.trs

    # Checks before run
    if check_same_cwsp(data, trs_list) is False:
        raise ValueError("The input tracers use different covariance workspaces")

    for trs in trs_list:
        if len(trs) != 4:
            raise ValueError("A combination of tracers is not 4 items long")

    # Computation
    run_cwsp_batch(data, trs_list)
