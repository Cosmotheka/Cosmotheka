#!/usr/bin/python
import os
import random
import time
import numpy as np


def save_npz(fname, threshold=1e100, **kwargs):
    print(f"Saving {fname}")
    for kn, kv in kwargs.items():
        maxv = np.max(np.abs(kv))
        if maxv in (True, False):
            continue
        elif maxv is None:
            raise RuntimeError(f"Some values in {kn} are None. Stopping here")
        elif np.isnan(maxv) or (maxv > threshold):
            raise RuntimeError(f"Some values in {kn} are nan or "
                               f">{threshold} (e.g. {maxv}). Stopping here")

    np.savez_compressed(fname, **kwargs, threshold=threshold)


def save_wsp(wsp, fname):
    """
    Write a workspace or covariance workspce

    Parameters
    ----------
    wsp: NmtWorkspace or NmtCovarianceWorkspace
        Workspace or Covariance workspace to save
    fname: str
        Path to save the workspace to
    """
    # Sleep a random number of ms. Dirty trick to avoid writing the same file
    # twice
    time.sleep(random.random() * 0.01)
    # Recheck again in case other process has started writing it
    if os.path.isfile(fname):
        return

    try:
        print(f"Saving {fname}")
        wsp.write_to(fname)
    except RuntimeError as e:
        if ('Error writing' in str(e)) and os.path.isfile(fname):
            # Check that the file has been created and the error is not due to
            # other problem (e.g. the folder does not exist)
            print(f"Error writing {fname}. Probably two processes are writing "
                  "at the same time. Removing it, computing it again and "
                  "trying to save it.")

            os.remove(fname)
            time.sleep(random.random() * 0.01)
            if not os.path.isfile(fname):
                wsp.write_to(fname)
        else:
            raise e


def read_wsp(wsp, fname, **kwargs):
    """
    Read a workspace or covariance workspace and removes it if fails

    Parameters
    ----------
    wsp: NmtWorkspace or NmtCovarianceWorkspace
        Workspace or Covariance workspace to save
    fname: str
        Path to save the workspace to
    kwargs:
        Arguments accepted by the (cov)workspace read_from method.
    """
    # Recheck again in case other process has started writing it
    try:
        print(f"Reading {fname}")
        wsp.read_from(fname, **kwargs)
    except RuntimeError as e:
        if ('Error reading' in str(e)) and os.path.isfile(fname):
            print(f"Error reading {fname}. Removing it and computing it again")
            os.remove(fname)
            return

        raise e
