#!/usr/bin/python
import os


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
