#!/usr/bin/python

import os
import time
import common as co

##############################################################################
def get_mem(data, trs, compute):
    # Return memory for nside 4096
    d = {}
    if compute == 'cls':
        d[0] = 11
        d[2] = 25
    elif compute == 'cov':
        d[0] = 16
        d[2] = 47
    else:
        raise ValueError('{} not defined'.format(compute))

    mem = 0
    for tr in trs:
        s = data['tracers'][tr]['spin']
        mem += d[s]

    return mem


def launch_cls(data, queue, njobs, wsp=False):
    #######
    nc = 4
    #
    cl_tracers = co.get_cl_tracers(data, wsp)
    outdir = data['output']
    c = 0
    for tr1, tr2 in cl_tracers:
        if c >= njobs:
            break
        comment = 'cl_{}_{}'.format(tr1, tr2)
        # TODO: don't hard-code it!
        trreq = ''.join(s for s in (tr1 + '_' + tr2) if not s.isdigit())
        fname = os.path.join(outdir, trreq, comment + '.npz')
        if os.path.isfile(fname):
            continue

        mem = get_mem(data, (tr1, tr2), 'cls') / nc
        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = 'cl.py {} {} {}'.format(args.INPUT, tr1, tr2)
        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)

def launch_cov(data, queue, njobs, wsp=False):
    #######
    nc = 10
    mem = 5
    #
    cov_tracers = co.get_cov_tracers(data, wsp)
    outdir = data['output']
    c = 0
    for trs in cov_tracers:
        if c >= njobs:
            break
        comment = 'cov_{}_{}_{}_{}'.format(*trs)
        fname = os.path.join(outdir, 'cov', comment + '.npz')
        if os.path.isfile(fname):
            continue
        mem = get_mem(data, trs, 'cov') / nc
        pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
        pyrun = 'cov.py {} {} {} {} {}'.format(args.INPUT, *trs)
        print(pyexec + " " + pyrun)
        os.system(pyexec + " " + pyrun)
        c += 1
        time.sleep(1)

def launch_to_sacc(data, name, nl, queue):
    outdir = data['output']
    fname = os.path.join(outdir, name)
    if os.path.isfile(fname):
        return

    nc = 10
    mem = 10
    comment = 'to_sacc'
    pyexec = "addqueue -c {} -n 1x{} -s -q {} -m {} /usr/bin/python3".format(comment, nc, queue, mem)
    pyrun = 'to_sacc.py {} {}'.format(args.INPUT, name)
    if nl:
        pyrun += ' --use_nl'
    print(pyexec + " " + pyrun)
    os.system(pyexec + " " + pyrun)

##############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('compute', type=str, help='Compute: cls, cov or to_sacc.')
    parser.add_argument('--queue', type=str, default='berg', help='SLURM queue to use')
    parser.add_argument('--njobs', type=int, default=20, help='Maximum number of jobs to launch')
    parser.add_argument('--wsp', default=False, action='store_true',
                        help='Set if you want to compute the different workspaces first')
    parser.add_argument('--to_sacc_name', type=str, default='cls_cov.fits', help='Sacc file name')
    parser.add_argument('--to_sacc_use_nl', default=False, action='store_true',
                        help='Set if you want to use nl and covNG (if present) instead of cls and covG ')
    args = parser.parse_args()

    ##############################################################################

    data = co.read_data(args.INPUT)

    queue = args.queue
    njobs = args.njobs

    if args.compute == 'cls':
        launch_cls(data, queue, njobs, args.wsp)
    elif args.compute == 'cov':
        launch_cov(data, queue, njobs, args.wsp)
    elif args.compute == 'to_sacc':
        launch_to_sacc(data, args.to_sacc_name, args.to_sacc_use_nl, queue)
    else:
        raise ValueError("Compute value '{}' not understood".format(args.compute))
