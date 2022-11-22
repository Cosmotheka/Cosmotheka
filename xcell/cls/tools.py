#!/usr/bin/python
import os
import random
import time
import numpy as np


def save_npz(fname, **kwargs):
    for kv in kwargs.values():
        if np.any(np.isnan(kv)):
            raise RuntimeError("Some computed values are nan. "
                               "Stopping here")

    np.savez_compressed(fname, **kwargs)


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
        wsp.write_to(fname)
    except RuntimeError as e:
        if ('Error writing' in str(e)) and os.path.isfile(fname):
            # Check that the file has been created and the error is not due to
            # other problem (e.g. the folder does not exist)
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
        wsp.read_from(fname, **kwargs)
    except RuntimeError as e:
        if ('Error reading' in str(e)) and os.path.isfile(fname):
            os.remove(fname)
            return

        raise e
