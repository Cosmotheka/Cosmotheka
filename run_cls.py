#!/usr/bin/python
from xcell.cls.data import Data
import os
import time
import subprocess
import numpy as np


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


def get_queued_jobs():
    result = subprocess.run(['q', '-tn'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


def check_skip(data, skip, trs):
    for tr in trs:
        if tr in skip:
            return True
        elif data.get_tracer_bare_name(tr) in skip:
            return True
    return False



def get_pyexec(comment, nc, queue, mem, onlogin, outdir):
    if onlogin:
        pyexec = "/usr/bin/python3"
    else:
        logdir = os.path.join(outdir, 'log')
        os.makedirs(logdir, exist_ok=True)
        logfname = os.path.join(logdir, comment + '.log')
        pyexec = "addqueue -o {} -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(logfname, comment, nc, queue, mem)

    return pyexec


def launch_cls(data, queue, njobs, nc, mem, fiducial=False, onlogin=False, skip=[]):
    #######
    #
    cl_tracers = data.get_cl_trs_names(wsp=True)
    cl_tracers += data.get_cl_trs_names(wsp=False)
    # Remove duplicates
    cl_tracers = np.unique(cl_tracers, axis=0).tolist()
    outdir = data.data['output']
    if fiducial:
        outdir = os.path.join(outdir, 'fiducial')

    if os.uname()[1] == 'glamdring':
        qjobs = get_queued_jobs()
    else:
        qjobs = ''

    c = 0
    for tr1, tr2 in cl_tracers:
        comment = 'cl_{}_{}'.format(tr1, tr2)
        if c >= njobs:
            break
        elif comment in qjobs:
            continue
        elif check_skip(data, skip, [tr1, tr2]):
            continue
        # TODO: don't hard-code it!
        trreq = data.get_tracers_bare_name_pair(tr1, tr2, '_')
        fname = os.path.join(outdir, trreq, comment + '.npz')
        recompute_cls = data.data['recompute']['cls']
        recompute_mcm = data.data['recompute']['mcm']
        recompute = recompute_cls or recompute_mcm
        if os.path.isfile(fname) and (not recompute):
            continue

        if not fiducial:
            pyexec = get_pyexec(comment, nc, queue, mem, onlogin, outdir)
            pyrun = '-m xcell.cls.cl {} {} {}'.format(args.INPUT, tr1, tr2)
        else:
            pyexec = get_pyexec(comment, nc, queue, 2, onlogin, outdir)
            pyrun = '-m xcell.cls.cl {} {} {} --fiducial'.format(args.INPUT, tr1, tr2)

        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)


def launch_cov(data, queue, njobs, nc, mem, onlogin=False, skip=[]):
    #######
    #
    cov_tracers = data.get_cov_trs_names(wsp=True)
    cov_tracers += data.get_cov_trs_names(wsp=False)
    cov_tracers = np.unique(cov_tracers, axis=0).tolist()
    outdir = data.data['output']

    if os.uname()[1] == 'glamdring':
        qjobs = get_queued_jobs()
    else:
        qjobs = ''

    c = 0
    for trs in cov_tracers:
        comment = 'cov_{}_{}_{}_{}'.format(*trs)
        if c >= njobs:
            break
        elif comment in qjobs:
            continue
        elif check_skip(data, skip, trs):
            continue
        fname = os.path.join(outdir, 'cov', comment + '.npz')
        recompute = data.data['recompute']['cov'] or data.data['recompute']['cmcm']
        if os.path.isfile(fname) and (not recompute):
            continue
        pyexec = get_pyexec(comment, nc, queue, mem, onlogin, outdir)
        pyrun = '-m xcell.cls.cov {} {} {} {} {}'.format(args.INPUT, *trs)
        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)


def launch_to_sacc(data, name, use, queue, nc, mem, onlogin=False):
    outdir = data.data['output']
    fname = os.path.join(outdir, name)
    if os.path.isfile(fname):
        return

    comment = 'to_sacc'
    pyexec = get_pyexec(comment, nc, queue, mem, onlogin, outdir)
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
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('compute', type=str, help='Compute: cls, cov or to_sacc.')
    parser.add_argument('-n', '--nc', type=int, default=28, help='Number of cores to use')
    parser.add_argument('-m', '--mem', type=int, default=7., help='Memory (in GB) per core to use')
    parser.add_argument('-q', '--queue', type=str, default='berg', help='SLURM queue to use')
    parser.add_argument('-j', '--njobs', type=int, default=100000, help='Maximum number of jobs to launch')
    parser.add_argument('--to_sacc_name', type=str, default='cls_cov.fits', help='Sacc file name')
    parser.add_argument('--to_sacc_use_nl', default=False, action='store_true',
                        help='Set if you want to use nl and cov extra (if present) instead of cls and covG ')
    parser.add_argument('--to_sacc_use_fiducial', default=False, action='store_true',
                        help="Set if you want to use the fiducial Cl and covG instead of data cls")
    parser.add_argument('--cls_fiducial', default=False, action='store_true', help='Set to compute the fiducial cls')
    parser.add_argument('--onlogin', default=False, action='store_true', help='Run the jobs in the login screen instead appending them to the queue')
    parser.add_argument('--skip', default=[], nargs='+', help='Skip the following tracers. It can be given as DELS__0 to skip only DELS__0 tracer or DELS to skip all DELS tracers')
    parser.add_argument('--override_yaml', default=False, action='store_true', help='Override the YAML file if already stored. Be ware that this could cause compatibility problems in your data!')
    args = parser.parse_args()

    ##############################################################################

    data = Data(data_path=args.INPUT, override=args.override_yaml)

    queue = args.queue
    njobs = args.njobs
    onlogin = args.onlogin

    if args.compute == 'cls':
        launch_cls(data, queue, njobs, args.nc, args.mem, args.cls_fiducial, onlogin, args.skip)
    elif args.compute == 'cov':
        launch_cov(data, queue, njobs, args.nc, args.mem, onlogin, args.skip)
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
        launch_to_sacc(data, args.to_sacc_name, use, queue, args.nc, args.mem, onlogin)
    else:
        raise ValueError(
                "Compute value '{}' not understood".format(args.compute))
