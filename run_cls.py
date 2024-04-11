#!/usr/bin/python
from cosmotheka.cls.data import Data
import os
import time
import subprocess
import numpy as np
import re
from datetime import datetime
from glob import glob


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


def get_pyexec(comment, nc, queue, mem, onlogin, outdir, batches=False, logfname=None):
    if batches:
        pyexec = ""
    else:
        pyexec = "/usr/local/shared/python3.9.7/bin/python3"

    if not onlogin:
        logdir = os.path.join(outdir, 'log')
        os.makedirs(logdir, exist_ok=True)
        if logfname is None:
            logfname = os.path.join(logdir, comment + '.log')
        pyexec = "addqueue -o {} -c {} -n 1x{} -s -q {} -m {} {}".format(logfname, comment, nc, queue, mem, pyexec)

    return pyexec


def get_jobs_with_same_cwsp(data):
    cov_tracers = data.get_cov_trs_names(wsp=True)
    cov_tracers += data.get_cov_trs_names(wsp=False)
    cov_tracers = np.unique(cov_tracers, axis=0).tolist()

    nsteps = len(cov_tracers)
    cwsp = {}
    for i, trs in enumerate(cov_tracers):
        # print(f"Splitting jobs in batches. Step {i}/{nsteps}")
        # Instantiating the Cov class is too slow
        # cov = Cov(data.data, *trs)
        # fname = cov.get_cwsp_path()

        # The problem with this is that any change in cov.py will not be seen here
        mask1, mask2, mask3, mask4 = [data.data['tracers'][trsi]["mask_name"] for trsi in trs]
        fname = os.path.join(data.data['output'],
                             f'cov/cw__{mask1}__{mask2}__{mask3}__{mask4}.fits')
        if fname in cwsp:
            cwsp[fname].append(trs)
        else:
            cwsp[fname] = [trs]

    return cwsp


def clean_lock(data):
    qjobs = get_queued_jobs()
    # regexp without the path since it does not appear when in queuing
    batches = re.findall('batch.*.sh', qjobs)

    batches_dir = os.path.join(data.data['output'], "run_batches")
    rm_lock_files = glob(os.path.join(batches_dir, "*_remove_locks.sh"))

    # Remove from rm_lock_files those that are still running
    rm_lf_tokeep = []
    for batch in batches:
        name = os.path.basename(batch).replace('.sh', '')
        for rm_lf in rm_lock_files.copy():
            if name in rm_lf:
                rm_lock_files.remove(rm_lf)
                rm_lf_tokeep.append(rm_lf)

    # Run the scripts of the processes that failed
    for rm_lf in rm_lock_files:
        print(f"Running cleaning script: {rm_lf}")
        os.system(f"/bin/bash {rm_lf}")
        os.remove(rm_lf)

    # Clean lock files of jobs that did not get to create the rm files
    cov_dir = os.path.join(data.data['output'], "cov")
    lock_files = glob(os.path.join(cov_dir, "*.lock"))
    for rm_lf in rm_lf_tokeep:
        with open(rm_lf, 'r') as f:
            for l in f.readlines():
                # [3:] to remove the rm
                if 'rm' not in l:
                    continue
                fname = re.search("rm .*.lock", l).group()[3:]
                # TODO: maybe a try it's faster
                try:
                    lock_files.remove(fname)
                except ValueError as e:
                    if 'not in list' in str(e):
                        continue
                    raise e

    for lf in lock_files:
        os.remove(lf)
    print("Finished cleaning lock files")


def launch_cov_batches(data, queue, njobs, nc, mem, onlogin=False, skip=[],
                        remove_cwsp=False, nnodes=1):
    def create_lock_file(fname):
        with open(fname + '.lock', 'w') as f:
            f.write('')

    def write_cw_sh(cw, covs_tbc):
        # Create a temporal file so that this cw script is not run elsewhere
        # This will be removed once the script finishes (independently if it
        # successes or fails)
        create_lock_file(cw)
        s = f"echo Running {cw}\n"
        s += f"/usr/bin/python3.8 run_cwsp_batch.py {args.INPUT}"
        if remove_cwsp:
            s += " --no_save_cw"
        for covi in covs_tbc:
            s += " -trs {} {} {} {}".format(*covi)
        s += "\n\n"

        s += f"echo Removing lock file: {cw}.lock\n"
        s += f'rm {cw}.lock\n\n'
        s += f"echo Finished running {cw}\n\n"
        return s

    outdir = data.data['output']
    cwsp = get_jobs_with_same_cwsp(data)

    # Create a folder to place the batch scripts. This is because I couldn't
    # figure out how to pass it through STDIN
    outdir_batches = os.path.join(outdir, 'run_batches')
    logfolder = os.path.join(outdir, 'log')
    os.makedirs(outdir_batches, exist_ok=True)

    # Create the covariance output directory
    covdir = os.path.join(outdir, "cov")
    os.makedirs(covdir, exist_ok=True)

    c = 0
    cw_tbc = []
    sh_tbc = []
    for ni, (cw, trs_list) in enumerate(cwsp.items()):
        if c >= njobs * nnodes:
            break
        elif os.path.isfile(cw + '.lock'):
            continue

        # Find the covariances to be computed
        covs_tbc = []
        for trs in trs_list:
            fname = os.path.join(covdir, 'cov_{}_{}_{}_{}.npz'.format(*trs))
            recompute = data.data['recompute']['cov'] or data.data['recompute']['cmcm']
            if (os.path.isfile(fname) and (not recompute)) or \
                    check_skip(data, skip, trs):
                continue

            covs_tbc.append(trs)

        # To avoid writing and launching an empty file (which will fail if
        # remove_cwsp is True when it tries to remove the cw.
        if len(covs_tbc) == 0:
            continue

        cw_tbc.append(cw)
        sh_tbc.append(write_cw_sh(cw, covs_tbc))
        c += 1

    n_total_jobs = len(cw_tbc)
    date = datetime.utcnow()
    timestamp = date.strftime("%Y%m%d%H%M%S")
    for nodei in range(nnodes):
        if njobs > n_total_jobs / nnodes:
            njobs = int(n_total_jobs / nnodes + 1)
        # ~1min per job for nside 1024
        time_expected = njobs / (60 * 24)
        name = f"batch{nodei}-{njobs}_{timestamp}"
        comment = f"{name}\(~{time_expected:.1f}days\)"
        sh_name = os.path.join(outdir_batches, f'{name}.sh')
        rm_name = os.path.join(outdir_batches, f'{name}_remove_locks.sh')
        logfname = os.path.join(logfolder, f'{name}.sh.log')

        # Create the script that will be launched
        with open(sh_name, 'w') as f:
            f.write('#!/bin/bash\n\n')
            for i, shi in enumerate(sh_tbc[nodei * njobs : (nodei + 1) * njobs], 1):
                f.write(f"# Step {i} / {njobs}\n")
                f.write(f"echo Step {i} / {njobs}\n")
                f.write(shi)
            f.write("echo Removing cleaning script\n")
            f.write(f"rm {rm_name}")

        # Create a file that will be used to remove the orphan lock files
        with open(rm_name, 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write('echo Removing lock files\n')
            for cwi in cw_tbc[nodei * njobs : (nodei + 1) * njobs]:
                f.write(f"[ -f {cwi}.lock ] && rm {cwi}.lock\n")
            f.write('echo Finished removing lock files\n')

        pyexec = get_pyexec(comment, nc, queue, mem, onlogin, outdir,
                            batches=True, logfname=logfname)
        print("##################################")
        print(pyexec + " " + sh_name)
        print("##################################")
        print()
        os.system(f"chmod +x {sh_name}")
        os.system(pyexec + " " + sh_name)
        time.sleep(1)


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
            pyrun = '-m cosmotheka.cls.cl {} {} {}'.format(args.INPUT, tr1, tr2)
        else:
            pyexec = get_pyexec(comment, nc, queue, 2, onlogin, outdir)
            pyrun = '-m cosmotheka.cls.cl {} {} {} --fiducial'.format(args.INPUT, tr1, tr2)

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
        pyrun = '-m cosmotheka.cls.cov {} {} {} {} {}'.format(args.INPUT, *trs)
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
    pyrun = '-m cosmotheka.cls.to_sacc {} {}'.format(args.INPUT, name)
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
    parser.add_argument('-N', '--nnodes', type=int, default=1, help='Number of nodes to use. If given, the jobs will be launched all in the same node')
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
    parser.add_argument('--batches', default=False, action='store_true',
                        help='Run the covariances in batches with all the ' +
                        'blocks sharing the same covariance workspace in a ' +
                        'single job')
    parser.add_argument('--remove_cwsp', default=False, action='store_true',
                        help='Remove the covariance workspace once the ' +
                        'batch job has finished')
    parser.add_argument('--clean_lock', default=False, action="store_true",
                        help="Remove lock files from failed runs.")
    args = parser.parse_args()

    ##############################################################################

    data = Data(data_path=args.INPUT, override=args.override_yaml)

    queue = args.queue
    njobs = args.njobs
    onlogin = args.onlogin
    nnodes = args.nnodes


    if args.clean_lock:
        clean_lock(data)
    elif args.compute == 'cls':
        launch_cls(data, queue, njobs, args.nc, args.mem, args.cls_fiducial, onlogin, args.skip)
    elif args.compute == 'cov':
        if args.batches:
            launch_cov_batches(data, queue, njobs, args.nc, args.mem, onlogin,
                               args.skip, args.remove_cwsp, nnodes)
        else:
            launch_cov(data, queue, njobs, args.nc, args.mem, onlogin,
                       args.skip)
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
