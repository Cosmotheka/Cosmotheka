#!/usr/bin/python
from xcell.cls.common import Data
import os
import time


##############################################################################
def get_mem(data, trs, compute):
    # Return memory for nside 4096
    d = {}
    if compute == 'cls':
        d[0] = 16
        d[2] = 25
    elif compute == 'cov':
        d[0] = 16
        d[2] = 47
    else:
        raise ValueError('{} not defined'.format(compute))

    mem = 0
    for tr in trs:
        mapper = data.get_mapper(tr)
        s = mapper.get_spin()
        mem += d[s]

    return mem


def launch_cls(data, queue, njobs, nc, mem, wsp=False, fiducial=False):
    #######
    #
    cl_tracers = data.get_cl_trs_names(wsp)
    outdir = data.data['output']
    if fiducial:
        outdir = os.path.join(outdir, 'fiducial')
    c = 0
    for tr1, tr2 in cl_tracers:
        if c >= njobs:
            break
        comment = 'cl_{}_{}'.format(tr1, tr2)
        # TODO: don't hard-code it!
        trreq = data.get_tracers_bare_name_pair(tr1, tr2, '_')
        fname = os.path.join(outdir, trreq, comment + '.npz')
        if os.path.isfile(fname):
            continue

        if not fiducial:
            pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
            pyrun = '-m xcell.cls.cl {} {} {}'.format(args.INPUT, tr1, tr2)
        else:
            pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, 2)
            pyrun = '-m xcell.cls.cl {} {} {} --fiducial'.format(args.INPUT, tr1, tr2)

        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)


def launch_cov(data, queue, njobs, nc, mem, wsp=False):
    #######
    #
    cov_tracers = data.get_cov_trs_names(wsp)
    outdir = data.data['output']
    c = 0
    for trs in cov_tracers:
        if c >= njobs:
            break
        comment = 'cov_{}_{}_{}_{}'.format(*trs)
        fname = os.path.join(outdir, 'cov', comment + '.npz')
        if os.path.isfile(fname):
            continue
        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = '-m xcell.cls.cov {} {} {} {} {}'.format(args.INPUT, *trs)
        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)


def launch_to_sacc(data, name, use, queue, nc, mem):
    outdir = data.data['output']
    fname = os.path.join(outdir, name)
    if os.path.isfile(fname):
        return

    comment = 'to_sacc'
    pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
    pyrun = '-m xcell.cls.to_sacc {} {}'.format(args.INPUT, name)
    if use == 'nl':
        pyrun += ' --use_nl'
    elif use == 'fiducial':
        pyrun += ' --use_fiducial'
    print(pyexec + " " + pyrun)
    os.system(pyexec + " " + pyrun)

##############################################################################


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('compute', type=str, help='Compute: cls, cov or to_sacc.')
    parser.add_argument('--nc', type=int, default=28, help='Maximum number of jobs to launch')
    parser.add_argument('--mem', type=int, default=7., help='Maximum number of jobs to launch')
    parser.add_argument('--queue', type=str, default='berg', help='SLURM queue to use')
    parser.add_argument('--njobs', type=int, default=100000, help='Maximum number of jobs to launch')
    parser.add_argument('--wsp', default=False, action='store_true',
                        help='Set if you want to compute the different workspaces first')
    parser.add_argument('--to_sacc_name', type=str, default='cls_cov.fits', help='Sacc file name')
    parser.add_argument('--to_sacc_use_nl', default=False, action='store_true',
                        help='Set if you want to use nl and covNG (if present) instead of cls and covG ')
    parser.add_argument('--to_sacc_use_fiducial', default=False, action='store_true',
                        help="Set if you want to use the fiducial Cl and covG instead of data cls")
    parser.add_argument('--cls_fiducial', default=False, action='store_true', help='Set to compute the fiducial cls')
    args = parser.parse_args()

    ##############################################################################

    data = Data(data_path=args.INPUT)

    queue = args.queue
    njobs = args.njobs

    if args.compute == 'cls':
        launch_cls(data, queue, njobs, args.nc, args.mem, args.wsp, args.cls_fiducial)
    elif args.compute == 'cov':
        launch_cov(data, queue, njobs, args.nc, args.mem, args.wsp)
    elif args.compute == 'to_sacc':
        if args.to_sacc_use_nl and args.to_sacc_use_fiducial:
            raise ValueError(
                    'Only one of --to_sacc_use_nl or --to_sacc_use_fiducial can be set')
        elif args.to_sacc_use_nl:
            use = 'nl'
        elif args.to_sacc_use_fiducial:
            use = 'fiducial'
        else:
            use = 'cls'
        launch_to_sacc(data, args.to_sacc_name, use, queue, args.nc, args.mem)
    else:
        raise ValueError(
                "Compute value '{}' not understood".format(args.compute))
